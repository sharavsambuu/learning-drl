#
# HAPPY TEXT GENERATOR WITH GRPO (TOY LLM HELLO WORLD)
#
# Зорилго
# Энэхүү код нь орчин үеийн LLM буюу хэлний том моделийн сургалтын арга барил болох 
# GRPO алгоритмыг энгийн түвшинд хэрэгжүүлж үзэх ингэхдээ TinyStories өгөгдөл дээр 
# суурилан англи хэл сурч улмаар GRPO ашиглан үргэлж эерэг аз жаргалтай түүх зохиодог 
# Happy текст үүсгүүр бүтээх явдал юм
#
# Архитектур
# Backbone нь 2 давхаргат LSTM сүлжээний хамтаар
# Global Attention буюу тэмдэгт бүр өмнөх түүхээ харах боломжтой бүтэц
#
# Сургалтын үе шатууд
# PHASE 1 буюу SFT (Supervised Fine-Tuning)
# Модель үсэг үг өгүүлбэр зүйг сурна
# Зорилго нь хэлний дүрмийн хувьд зөв текст бичдэг болох
#
# PHASE 2 буюу GRPO (Alignment Reinforcement Learning)
# Модель Happy байх шагнал буюу Reward ын төлөө суралцана
# Critic модель ашиглахгүйгээр Memory Efficient байдлаар өөрийн үүсгэсэн олон хувилбарууд 
# дундаас шилдэгийг нь сонгох замаар бодлогоо шинэчилнэ
#
# Тохиргоо 
# 12GB GPU Optimized
#

import os
# JAX ийн санах ойн хуваарилалтыг хязгаарлах буюу OOM алдаанаас сэргийлэх тохиргоо
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"
import re
import math
import random
import jax
import optax
import numpy         as np
import flax.linen    as nn
from   jax           import numpy as jnp
from   flax.training import train_state


debug                 = True
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42


# HYPERPARAMETERS (12GB VRAM зориулсан тохиргоо)

# SFT буюу Supervised Fine-Tuning үе шатны тохиргоо
sft_total_steps       = 5000
sft_batch_size        = 64      # GPU тооцооллох хэсгийг завгүй байлгана 
sft_seq_len           = 256     # 12GB VRAM д багтах аюулгүй урт
sft_learning_rate     = 0.0015
sft_warmup_steps      = 400
sft_sample_freq       = 500     # Хэдэн алхам тутамд SFT үр дүнг хэвлэж харах вэ

# GRPO буюу Reinforcement Learning үе шатны тохиргоо
grpo_total_updates    = 1500
prompts_per_update    = 4       # Нэг алхамд 4 prompt үржих нь 16 group тэнцүү 64 rollout буюу Batch тай ижил
group_size            = 16      # GRPO ийн харьцуулалт хийх бүлгийн хэмжээ
gen_len               = 256     # Үүсгэх текстийн урт SFT тэй ижил
grpo_temp             = 1.0     # Текст үүсгэх үеийн санамсаргүй байдал
grpo_sample_freq      = 20      # Хэдэн update тутамд үр дүнг хэвлэж харах вэ

# PPO буюу Proximal Policy Optimization тохиргоо
ppo_epochs            = 3
mini_batch_size       = 64      # Update хийх хурдыг дэмжинэ
clip_epsilon          = 0.2
entropy_coeff         = 0.02    # Моделийг хэт нэг хэвийн болохоос сэргийлнэ
grpo_lr               = 0.0004
max_grad_norm         = 0.5

# Dynamic KL Divergence буюу Модель галзуурахаас сэргийлэх тохиргоо
kl_beta               = 0.04    # Анхны шийтгэлийн хэмжээ
target_kl             = 0.05    # Зорилтот өөрчлөлтийн хэмжээ
kl_alpha              = 1.2     # Beta-г өөрчлөх хурд

# Моделийн дотоод бүтцийн тохиргоо
prompt_len            = 48
model_max_len         = 320     # 256 нэмэх нь 48 тэнцүү 304 тул нөөцтэйгөөр 320
num_lstm_layers       = 2
embed_dim             = 32      # Тэмдэгт учраас бага байхад болно
hidden_dim            = 128     # Тооцооллыг хөнгөлж Batch Size ийг нэмэх боломж олгоно

# Эерэг болон сөрөг үгсийн сан
happy_vocab = ["happy", "joy", "joyful", "smile", "smiled", "laugh", "laughed", "love", "loved", "kind", "nice", "fun", "good", "great", "amazing", "wonderful", "excited", "brave", "bright", "safe", "friend", "friends"]
sad_vocab   = ["sad", "cry", "cried", "bad", "angry", "mad", "hurt", "scary", "afraid", "fear", "dark", "hate", "hated", "mean", "alone", "lost", "dead", "death", "kill", "killed"]
negations   = ["not", "no", "never", "don't", "can't", "won't", "neither", "nor"]

np.random.seed(seed)
random.seed(seed)


# DATASET AND TOKENIZATION

# Файлаас унших хэсэг
if not os.path.exists(dataset_path):
    print(f"Анхаар {dataset_path} файл олдсонгүй Жишээ текст ашиглана")
    raw_text = "Once upon a time there was a happy robot. It loved to smile. " * 1000
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

# Түүхүүдийг задлах буюу SFT сургалтад бэлдэх
all_stories = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 0]
# GRPO шатанд ашиглах эерэг түүхүүдийн дээж буюу reference болгож ашиглана
positive_stories = [s for s in all_stories if any(w in s.lower() for w in happy_vocab)]
if not positive_stories: positive_stories = all_stories

# Тэмдэгт болон ID хөрвүүлэлт хийх Character Level Tokenizer
unique_chars   = sorted(list(set("".join(all_stories))))
PAD, BOS       = "<PAD>", "<BOS>"
chars          = [PAD, BOS] + unique_chars
char_to_id     = {c: i for i, c in enumerate(chars)}
id_to_char     = {i: c for c, i in char_to_id.items()}
vocab_size     = len(chars)
pad_id, bos_id = char_to_id[PAD], char_to_id[BOS]

def encode_text(text):
    return [bos_id] + [char_to_id.get(ch, pad_id) for ch in text]

def decode_ids(ids):
    return "".join([id_to_char[int(i)] for i in ids if int(i) not in [pad_id, bos_id]])

corpus_text = "\n\n".join(positive_stories)
corpus_ids  = np.array(encode_text(corpus_text), dtype=np.int32)

print(f"Vocab Size: {vocab_size}")
print(f"Total Chars: {len(corpus_ids)}")


# MODEL ARCHITECTURE (TinyLLM)

class TinyLLM(nn.Module):
    vocab_size: int; embed_dim: int; hidden_dim: int; num_layers: int; max_len: int

    @nn.compact
    def __call__(self, token_id, carry, deterministic=True):
        # carry нь lstm states болон attention memory мөн current step index агуулна
        lstm_carries, attn_mem, step_idx = carry
        x = nn.Embed(self.vocab_size, self.embed_dim)(token_id)

        # LSTM Layers буюу дараалсан мэдээллийг боловсруулах хэсэг
        new_carries = []
        for i in range(self.num_layers):
            c_i, h_i      = lstm_carries[i]
            (c_i, h_i), h = nn.LSTMCell(self.hidden_dim, name=f"lstm_{i}")((c_i, h_i), x)
            new_carries.append((c_i, h))
            x = h  # LSTM ийн гаралтыг дараагийн давхарга руу дамжуулна

        # Global Attention Update буюу санах ойг шинэчлэх хэсэг
        # Одоогийн LSTM ийн гаралтыг санах ойн step_idx байрлалд бичнэ
        h_expanded = x[:, None, :]
        attn_mem   = jax.lax.dynamic_update_slice(attn_mem, h_expanded, (0, step_idx, 0))

        # Attention буюу өнгөрсөн бүх мэдээлэл рүү хандах хэсэг
        q = nn.Dense(self.hidden_dim, name="attn_q")(x)
        k = nn.Dense(self.hidden_dim, name="attn_k")(attn_mem)
        v = nn.Dense(self.hidden_dim, name="attn_v")(attn_mem)

        scores = jnp.sum(k * q[:, None, :], axis=-1) / jnp.sqrt(self.hidden_dim)
        # Masking буюу ирээдүйн хараахан бичигдээгүй хэсгийг харахгүй болгох
        mask   = jnp.arange(self.max_len) <= step_idx
        scores = jnp.where(mask[None, :], scores, -1e9)
        w      = jax.nn.softmax(scores, axis=-1)
        ctx    = jnp.sum(v * w[:, :, None], axis=1)

        # Output Projection буюу гаралтын хэсэг
        h_mix  = nn.Dense(self.hidden_dim)(jnp.concatenate([x, ctx], axis=-1))
        h_mix  = nn.LayerNorm()(nn.tanh(h_mix))
        logits = nn.Dense(self.vocab_size)(h_mix)

        return logits, (tuple(new_carries), attn_mem, step_idx + 1)

def init_carry(batch_size):
    # Эхлэлийн төлөвийг тэгээр дүүргэх
    carries  = tuple((jnp.zeros((batch_size, hidden_dim)), jnp.zeros((batch_size, hidden_dim))) for _ in range(num_lstm_layers))
    attn_mem = jnp.zeros((batch_size, model_max_len, hidden_dim))
    return (carries, attn_mem, 0)

# Моделийг цэнэглэн эхлүүлэх
model  = TinyLLM(vocab_size, embed_dim, hidden_dim, num_lstm_layers, model_max_len)
params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,), jnp.int32), init_carry(1), deterministic=True)["params"]


# JAX HELPER FUNCTIONS

@jax.jit
def logprob_from_logits(logits, actions):
    # Сонгосон токенуудын log probability-ийг тооцох
    return jnp.take_along_axis(jax.nn.log_softmax(logits, -1), actions[:, None], 1).squeeze(1)

@jax.jit
def kl_from_logits(logits_new, logits_ref):
    # Хоёр моделийн гаралтын зөрүүг буюу KL Divergence ийг хэмжих
    p_new = jax.nn.softmax(logits_new, -1)
    return jnp.sum(p_new * (jax.nn.log_softmax(logits_new, -1) - jax.nn.log_softmax(logits_ref, -1)), -1).mean()

@jax.jit
def unroll_logits_eval(params, token_seq):
    # Өгөгдсөн дарааллын дагуу моделийг ажиллуулж logits гаргах training mode
    def step_fn(carry, tok):
        logits, carry = model.apply({"params": params}, tok, carry, deterministic=True)
        return carry, logits
    _, logits_seq = jax.lax.scan(step_fn, init_carry(token_seq.shape[0]), token_seq.T)
    return jnp.transpose(logits_seq, (1, 0, 2))

@jax.jit
def generate_rollout(behavior_params, prompt_tokens, key, temperature):
    # Prompt хэсгийг уншуулж carry төлөвийг бэлдэх хэсэг
    def condition_step(carry, tok):
        logits, carry = model.apply({"params": behavior_params}, tok, carry, deterministic=True)
        return carry, logits
    carry, _ = jax.lax.scan(condition_step, init_carry(prompt_tokens.shape[0]), prompt_tokens.T)

    # Текст үргэлжлүүлэн үүсгэх буюу auto-regressive generation
    def gen_step(carry_key_tok, _):
        carry, key, tok = carry_key_tok
        logits, carry   = model.apply({"params": behavior_params}, tok, carry, deterministic=True)
        key, subkey     = jax.random.split(key)
        # Temperature sampling хийх
        next_tok        = jax.random.categorical(subkey, logits / temperature, axis=-1).astype(jnp.int32)
        return (carry, key, next_tok), (next_tok, logprob_from_logits(logits, next_tok))

    # Gen len урттай текст үүсгэх
    _, (gen_toks, gen_lps) = jax.lax.scan(gen_step, (carry, key, prompt_tokens[:, -1]), length=gen_len)
    return jnp.concatenate([prompt_tokens, gen_toks.T], axis=1), gen_lps.T


# REWARD FUNCTION (Rule-based)

def reward_hybrid_pro(text, fluency_score):
    """
    Текстийн чанар болон happy байдлыг үнэлэх функц
    """
    t     = text.lower()
    words = re.findall(r"[a-z']+", t)
    
    # Хэт богино текстэд хатуу шийтгэл өгөх
    if len(words) < 6: return -4.0

    score         = 0.0
    happy_matches = 0
    
    # Үгсийн утгыг шалгах буюу Semantic Check
    for i, w in enumerate(words):
        if w in happy_vocab:
            # Үгүйсгэл шалгах буюу Context Check, Жишээ нь not happy
            context = words[max(0, i-2):i]
            if any(n in context for n in negations):
                score -= 3.0
            else:
                score         += 2.5
                happy_matches += 1
        elif w in sad_vocab:
            score -= 2.0

    # Keyword Spamming Penalty буюу нэг үгийг хэт давтахаас сэргийлнэ
    # Жишээ нь happy happy happy гэж бичихээс сэргийлнэ
    for w in set(words):
        count = words.count(w)
        if count > 3: score -= (count - 3) * 1.0

    # Diversity Bonus буюу үгсийн баялаг байдлыг урамшуулах
    diversity = len(set(words)) / len(words)
    score    += diversity * 4.0

    # Fluency Gate буюу утгагүй зүйл бичихээс сэргийлнэ
    # Reference model ийн logprob хэт бага байвал шийтгэнэ
    if fluency_score < -3.5: score -= 4.0

    # Бүтцийн оноо өгөх
    if text.strip().endswith(('.', '!', '?')): score += 1.5
    if len(text) > 100: score += 1.0

    return max(-10.0, min(10.0, score))

def compute_grpo_advantages(rewards, n_prompts, g_size):
    """
    GRPO ийн гол логик буюу групп доторх харьцуулалт
    Prompt бүрийн хувьд үүсгэсэн 16 хувилбарын дунджийг олж
    түүнээс дээгүүр оноо авсныг нь Advantage гэж үзнэ
    """
    rg   = rewards.reshape(n_prompts, g_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    std  = np.std (rg, axis=1, keepdims=True) + 1e-8
    adv  = (rg - mean) / std
    return adv.reshape(-1).astype(np.float32), float(mean.mean())


# PHASE, 1 SFT (Supervised Fine-Tuning)

sft_tx    = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(optax.warmup_cosine_decay_schedule(0, sft_learning_rate, sft_warmup_steps, sft_total_steps)))
sft_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)

@jax.jit
def sft_step(state, batch):
    def loss_fn(p):
        logits = unroll_logits_eval(p, batch[:, :-1])
        mask   = (batch[:, 1:] != pad_id).astype(jnp.float32)
        loss   = optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:])
        return jnp.sum(loss * mask) / jnp.sum(mask)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

print("\n" + "="*50)
print("  PHASE 1: SFT - Суурь хэлний мэдлэг олгох")
print(f"  Steps: {sft_total_steps} | Batch: {sft_batch_size} | Seq: {sft_seq_len}")
print("="*50 + "\n")

for step in range(sft_total_steps):
    # Датасетээс санамсаргүй хэсгийг таслан авах
    starts = np.random.randint(0, corpus_ids.shape[0] - sft_seq_len - 1, sft_batch_size)
    batch  = np.stack([corpus_ids[s:s+sft_seq_len+1] for s in starts])
    
    # Машин сургалтын нэг алхам
    sft_state, sft_loss = sft_step(sft_state, jnp.asarray(batch))
    
    # Явцыг хэвлэх
    if step % 500 == 0:
        print(f"[SFT] Step {step:4d} | Loss: {sft_loss:.4f}")

    # Модель хэрхэн бичиж сурч байгааг харах хэсэг
    if step > 0 and step % sft_sample_freq == 0:
        # Тогтмол prompt ашиглан өөрчлөлтийг харьцуулах
        test_key    = jax.random.PRNGKey(step)
        test_prompt = jnp.full((1, prompt_len), pad_id, dtype=jnp.int32)
        # Once upon a time гэж эхлүүлэх
        start_phr   = encode_text("Once upon a time")[:prompt_len]
        test_prompt = test_prompt.at[0, :len(start_phr)].set(jnp.array(start_phr))
        
        sample, _   = generate_rollout(sft_state.params, test_prompt, test_key, 0.8)
        decoded     = decode_ids(sample[0, prompt_len:])
        print(f"   >> Sample: {decoded[:120]}...")


# PHASE 2, GRPO (Group Relative Policy Optimization)

grpo_state = train_state.TrainState.create(apply_fn=model.apply, params=sft_state.params, tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(grpo_lr)))
frozen_ref = grpo_state.params # Reference модель буюу SFT ээс сурсан мэдлэгээ хадгалах

@jax.jit
def grpo_step(state, ref_params, rollouts, old_lps, advs, beta):
    gen_start, gen_end = prompt_len - 1, prompt_len - 1 + gen_len
    def loss_fn(p):
        # Шинэ logits тооцох
        logits = unroll_logits_eval(p, rollouts[:, :-1])[:, gen_start:gen_end]
        logp   = logprob_from_logits(logits, rollouts[:, prompt_len:])
        
        # PPO Ratio буюу шинэ болон хуучин магадлалын харьцаа
        ratio  = jnp.exp(logp - old_lps)
        surr1  = ratio * advs[:, None]
        surr2  = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advs[:, None]
        
        # KL Divergence буюу Reference ээс хэт зөрөхгүй байх тохиргоо
        ref_logits = unroll_logits_eval(ref_params, rollouts[:, :-1])[:, gen_start:gen_end]
        kl         = kl_from_logits(logits.reshape((-1, vocab_size)), ref_logits.reshape((-1, vocab_size)))
        
        pg_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        return pg_loss + beta * kl, (pg_loss, kl)
    
    (loss, (pg, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, pg, kl

print("\n" + "="*50)
print("  PHASE 2: GRPO - Happy бодлого сургах")
print(f"  Updates: {grpo_total_updates} | Group: {group_size} | Prompts: {prompts_per_update}")
print("="*50 + "\n")

for update in range(grpo_total_updates):
    # Санамсаргүй prompt бэлдэх
    p_list = []
    for _ in range(prompts_per_update):
        story = random.choice(all_stories)
        ids   = encode_text(story)
        start = random.randint(0, max(0, len(ids) - prompt_len))
        p_ids = np.full((prompt_len,), pad_id, dtype=np.int32)
        chunk = ids[start:start+prompt_len]
        p_ids[:len(chunk)] = chunk
        p_list.append(p_ids)
    
    # Rollout Generation буюу Олон хувилбар үүсгэх
    prompts = np.repeat(np.stack(p_list), group_size, axis=0)
    key     = jax.random.PRNGKey(update)
    rollouts, behavior_lps = generate_rollout(grpo_state.params, jnp.asarray(prompts), key, grpo_temp)
    
    # Fluency Score буюу Reference моделиор шалгах
    ref_lps_all = unroll_logits_eval(frozen_ref, rollouts[:, :-1])
    target_lps  = logprob_from_logits(ref_lps_all[:, prompt_len-1:-1], rollouts[:, prompt_len:])
    fluency     = np.array(jnp.mean(target_lps, axis=1))
    
    # Reward Calculation болон Advantages тооцох
    rewards        = np.array([reward_hybrid_pro(decode_ids(rollouts[i, prompt_len:]), fluency[i]) for i in range(rollouts.shape[0])])
    advs, m_reward = compute_grpo_advantages(rewards, prompts_per_update, group_size)
    
    # PPO Update буюу LLM-ээ сайжруулж сургах алхам
    for _ in range(ppo_epochs):
        grpo_state, _, pg_l, kl_l = grpo_step(grpo_state, frozen_ref, rollouts, behavior_lps, jnp.asarray(advs), kl_beta)

    # Dynamic KL буюу Adaptive Controller
    if kl_l > target_kl * 1.5:
        kl_beta *= kl_alpha
    elif kl_l < target_kl / 1.5:
        kl_beta /= kl_alpha

    # Явц хэвлэж харах
    if update % 20 == 0:
        print(f"[GRPO] Upd {update:4d} | AvgReward: {m_reward:6.2f} | KL: {float(kl_l):.4f} | Beta: {kl_beta:.4f}")
    
    if update % grpo_sample_freq == 0:
        # Хамгийн өндөр оноо авсан жишээг харуулах
        best_idx = np.argmax(rewards)
        best_txt = decode_ids(rollouts[best_idx, prompt_len:])
        print(f"   >> Best Sample: {best_txt[:160]}...")

print("\n=== СУРГАЛТ ДУУСЛАА ===")
# Төгсгөлийн шалгалт
final_prompt = np.full((1, prompt_len), pad_id, dtype=np.int32)
final_prompt[0, :4] = encode_text("Once")[:4]
final_rollout, _ = generate_rollout(grpo_state.params, jnp.asarray(final_prompt), jax.random.PRNGKey(999), 0.8)
print(f"Final Result: {decode_ids(final_rollout[0, prompt_len:])}")