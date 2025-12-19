#
# HAPPY TEXT GENERATOR TRANSFORMER WITH GRPO TUNING
#
# Зорилго
# Энэхүү код нь орчин үеийн LLM ийг alignment хийхэд хэрэглэдэг сургалтын арга барил болох 
# GRPO алгоритмыг хамгийн хялбар түвшинд хэрэгжүүлж үзэх, ингэхдээ TinyStories өгөгдөл дээр 
# суурилан англи хэл сурч улмаар GRPO ашиглан үргэлж эерэг аз жаргалтай түүх зохиодог 
# байхаар happy текст үүсгүүр бүтээх явдал юм.
#
# Машин сургалтын үе шатууд
# PHASE 1 буюу SFT (Supervised Fine-Tuning)
#   Модель үсэг, үг бүтэх, өгүүлбэр зүй гэх мэтийг сурна.
#   Зорилго нь хэлний дүрмийн хувьд зөв текст бичдэг болох.
#
# PHASE 2 буюу GRPO (Alignment Reinforcement Learning)
#   Модель happy байх шагнал буюу Reward-ын төлөө суралцана.
#   Critic модель ашиглахгүйгээр Memory Efficient байдлаар өөрийн үүсгэсэн олон 
#   хувилбарууд дундаас шилдгийг нь сонгох замаар alignment хийнэ.
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


# HYPERPARAMETERS, RTX 5070 Ti 12GB VRAM зориулсан тохиргоо

# SFT буюу Supervised Fine-Tuning үе шатны тохиргоо
sft_total_steps       = 5000
sft_batch_size        = 128     
sft_seq_len           = 256     # Context window
sft_learning_rate     = 5e-4    
sft_warmup_steps      = 200
sft_sample_freq       = 500     # Хэдэн алхам тутамд SFT үр дүнг хэвлэж харах вэ

# GRPO буюу Reinforcement Learning үе шатны тохиргоо
grpo_total_updates    = 1500
group_size            = 16      # GRPO ийн харьцуулалт хийх бүлгийн хэмжээ
prompts_per_update    = 4       # Нэг алхамд 4 prompt үржихдэг нь 16 group буюу 64 rollout
gen_len               = 256     # Үүсгэх текстийн урт
grpo_temp             = 0.9     # Текст үүсгэх температур
grpo_sample_freq      = 20      # Хэдэн update тутамд үр дүнг хэвлэж харах вэ

# PPO буюу Proximal Policy Optimization тохиргоо
ppo_epochs            = 3
mini_batch_size       = 64      
clip_epsilon          = 0.2
entropy_coeff         = 0.01    
grpo_lr               = 2e-5    # Fine-tuning үед маш бага Learning Rate хэрэг болно
max_grad_norm         = 1.0

# Dynamic KL Divergence буюу Модель галзуурахаас сэргийлэх тохиргоо
kl_beta               = 0.04    # Анхны шийтгэлийн хэмжээ
target_kl             = 0.05    # Зорилтот өөрчлөлтийн хэмжээ
kl_alpha              = 1.2     # Beta-г өөрчлөх хурд

# Моделийн дотоод бүтцийн тохиргоо
prompt_len            = 48
model_max_len         = 320     # Positional Embedding-д хэрэгтэй
num_layers            = 4       # Transformer блокуудын тоо 
num_heads             = 4       # Multi-head attention heads
embed_dim             = 128     # Embedding size 
dropout_rate          = 0.1     # Overfitting-ээс сэргийлнэ

# Эерэг болон сөрөг үгсийн сан
happy_vocab = ["happy", "joy", "joyful", "smile", "smiled", "laugh", "laughed", "love", "loved", "kind", "nice", "fun", "good", "great", "amazing", "wonderful", "excited", "brave", "bright", "safe", "friend", "friends"]
sad_vocab   = ["sad", "cry", "cried", "bad", "angry", "mad", "hurt", "scary", "afraid", "fear", "dark", "hate", "hated", "mean", "alone", "lost", "dead", "death", "kill", "killed"]
negations   = ["not", "no", "never", "don't", "can't", "won't", "neither", "nor"]

np.random.seed(seed)
random.seed(seed)


# DATASET AND TOKENIZATION

if not os.path.exists(dataset_path):
    print(f"Анхаар {dataset_path} файл олдсонгүй Жишээ текст ашиглана")
    raw_text = "Once upon a time there was a happy robot. It loved to smile. " * 2000
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

print(f"Vocab Size  : {vocab_size}")
print(f"Total Chars : {len(corpus_ids)}")


# MODEL ARCHITECTURE, TinyTransformer

class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        head_dim = self.embed_dim // self.num_heads
        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(x)
        v = nn.Dense(self.embed_dim)(x)

        # Split heads
        q = q.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        k = k.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        v = v.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)

        # Attention score
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(head_dim)
        
        # Causal Masking (ирээдүйг харахгүй байх)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
            
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Гаралтыг нэгтгэх
        out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        out = out.reshape(x.shape[0], x.shape[1], self.embed_dim)
        return nn.Dense(self.embed_dim)(out)

class MLP(nn.Module):
    embed_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        return x

class Block(nn.Module):
    embed_dim: int
    num_heads: int
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Attention with Residual
        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(nn.LayerNorm()(x), mask, deterministic)
        # MLP with Residual
        x = x + MLP(self.embed_dim)(nn.LayerNorm()(x))
        return x

class TinyTransformer(nn.Module):
    vocab_size: int; embed_dim: int; num_layers: int; num_heads: int; max_len: int

    @nn.compact
    def __call__(self, tokens, deterministic=True):
        b, t = tokens.shape
        
        # Token & Positional Embeddings
        tok_emb = nn.Embed(self.vocab_size, self.embed_dim)(tokens)
        pos_emb = nn.Embed(self.max_len, self.embed_dim)(jnp.arange(t))
        x = tok_emb + pos_emb[None, :, :]
        
        # Causal Mask (lower triangular)
        mask = jnp.tril(jnp.ones((t, t))) == 1
        mask = mask[None, None, :, :] # [Batch, Head, Q, K]

        # Transformer Blocks
        for _ in range(self.num_layers):
            x = Block(self.embed_dim, self.num_heads)(x, mask, deterministic)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits


# Моделийг цэнэглэн эхлүүлэх
model  = TinyTransformer(vocab_size, embed_dim, num_layers, num_heads, model_max_len)
# Dummy input
dummy_in = jnp.zeros((1, sft_seq_len), dtype=jnp.int32)
params   = model.init(jax.random.PRNGKey(seed), dummy_in, deterministic=True)["params"]


# JAX HELPERS

@jax.jit
def logprob_from_logits(logits, actions):
    # Logits  хэлбэр нь [B, T, V] эсвэл [B, V] байж болно
    # Actions хэлбэр нь [B, T]    эсвэл [B] байна
    # сүүлийн хэмжээс (vocab) дээрээс сонгосон үйлдлийн магадлалыг авах хэрэгтэй (axis=-1)
    
    logp = jax.nn.log_softmax(logits, axis=-1)
    
    # Take along axis шаардлагаар actions-ийн хэмжээсийг өргөтгөх
    # [..., None] нь төгсгөлд нь 1 хэмжээс нэмнэ -> [B, T, 1] эсвэл [B, 1]
    selected_logp = jnp.take_along_axis(logp, actions[..., None], axis=-1)
    
    return selected_logp.squeeze(-1)

@jax.jit
def kl_from_logits(logits_new, logits_ref):
    # Хоёр моделийн гаралтын зөрүүг буюу KL Divergence ийг хэмжих
    p_new = jax.nn.softmax(logits_new, -1)
    return jnp.sum(p_new * (jax.nn.log_softmax(logits_new, -1) - jax.nn.log_softmax(logits_ref, -1)), -1).mean()

@jax.jit
def unroll_logits_eval(params, token_seq):
    # Transformer дээр unroll хийнэ гэдэг нь зүгээр л шууд дуудаад ажиллуулах явдал юм
    return model.apply({"params": params}, token_seq, deterministic=True)

@jax.jit
def generate_rollout(behavior_params, prompt_tokens, key, temperature):
    """
    Transformer Generation Loop
    KV-Cache ашиглаагүй Toy Model учир давталт бүрт context-ийг дахин тооцоолно.
    """
    B, P = prompt_tokens.shape
    
    # Текст үүсгэх давталт - Loop state
    # current_seq: Одоо байгаа бүх текст (padding-тай)
    final_seq_len = P + gen_len
    # Эхлээд Prompt-ийг оруулаад үлдсэнийг Pad хийнэ
    current_seq = jnp.pad(prompt_tokens, ((0,0), (0, gen_len)), constant_values=pad_id)
    
    def scan_body(carry, i):
        seq, k = carry
        
        # Моделийг ажиллуулах
        logits = model.apply({"params": behavior_params}, seq, deterministic=True)
        
        # Дараагийн таамаглах ёстой индекс: PromptLen + i - 1
        # Жишээ нь: Prompt 48 урттай. i=0 үед 47-р индексийн гаралт нь 48-р үгийг таана.
        pred_idx = P + i - 1
        pred_logits = logits[:, pred_idx, :] # [Batch, Vocab]
        
        k, sk    = jax.random.split(k)
        next_tok = jax.random.categorical(sk, pred_logits / temperature).astype(jnp.int32)
        next_lp  = logprob_from_logits(pred_logits, next_tok)
        
        # Дарааллыг шинэчлэх (PromptLen + i байрлалд бичих)
        write_idx   = P + i
        new_tok_col = next_tok[:, None]
        seq         = jax.lax.dynamic_update_slice(seq, new_tok_col, (0, write_idx))
        
        return (seq, k), (next_tok, next_lp)

    # Run scan
    (final_seq, _), (gen_toks, gen_lps) = jax.lax.scan(scan_body, (current_seq, key), jnp.arange(gen_len))
    
    return final_seq, gen_lps.T


# REWARD FUNCTION, Rule-based

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
    """
    rg   = rewards.reshape(n_prompts, g_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    std  = np.std (rg, axis=1, keepdims=True) + 1e-8
    adv  = (rg - mean) / std
    return adv.reshape(-1).astype(np.float32), float(mean.mean())


# PHASE 1, SFT (Supervised Fine-Tuning)

sft_tx    = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adamw(optax.warmup_cosine_decay_schedule(0, sft_learning_rate, sft_warmup_steps, sft_total_steps), weight_decay=1e-4))
sft_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)

@jax.jit
def sft_step(state, batch):
    def loss_fn(p):
        # Transformer: Input = batch, Target = Shifted Batch
        logits = unroll_logits_eval(p, batch) # [B, T, V]
        
        # Shift targets: Input "A B C", Label "B C D"
        logits_trunc = logits[:, :-1, :]
        labels       = batch [:, 1:    ]
        
        mask   = (labels != pad_id).astype(jnp.float32)
        loss   = optax.softmax_cross_entropy_with_integer_labels(logits_trunc, labels)
        return jnp.sum(loss * mask) / jnp.sum(mask)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

print("\n" + "="*50)
print("  PHASE 1: SFT - Суурь хэлний мэдлэг олгох (Transformer)")
print(f"  Steps: {sft_total_steps} | Batch: {sft_batch_size} | Device: GPU (JIT)")
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
        # Pad-ийг хасаж хэвлэх
        decoded     = decode_ids(sample[0])
        # Prompt хэсгийг алгасаж харуулах (Transformer output нь full seq байдаг)
        print(f"   >> Sample: {decoded[len(decode_ids(test_prompt[0])):][:120]}...")


# PHASE 2, GRPO (Group Relative Policy Optimization)

grpo_state = train_state.TrainState.create(apply_fn=model.apply, params=sft_state.params, tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(grpo_lr)))
frozen_ref = grpo_state.params # Reference модель буюу SFT ээс сурсан мэдлэгээ хадгалах

@jax.jit
def grpo_step(state, ref_params, rollouts, old_lps, advs, beta):
    # Transformer дээр rollouts нь бүтэн [Prompt + Gen] байна
    # зөвхөн Gen хэсгийн логит дээр loss тооцно.
    gen_start = prompt_len
    
    def loss_fn(p):
        # Шинэ logits тооцох (Бүтэн seq дээр гүйлгэнэ)
        logits_all = unroll_logits_eval(p, rollouts) # [B, T, V]
        
        # Prompt-ийн төгсгөлөөс эхлээд generation-ий төгсгөл хүртэлх таамаглах
        # i-р индекс дээрхи logit i+1 ийг таана
        # rollouts[:, prompt_len:] буюу үүсгэсэн токенуудын магадлалыг хайна
        # эдгээр нь logits[:, prompt_len-1 : -1] дотор байгаа
        
        logits_gen = logits_all[:, gen_start-1:-1, :]
        logp       = logprob_from_logits(logits_gen, rollouts[:, gen_start:])
        
        # PPO Ratio буюу шинэ болон хуучин магадлалын харьцаа
        # old_lps нь generate_rollout-аас ирсэн (зөвхөн gen part)
        ratio  = jnp.exp(logp - old_lps)
        surr1  = ratio * advs[:, None]
        surr2  = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advs[:, None]
        
        # KL Divergence буюу Reference ээс хэт зөрөхгүй байх тохиргоо
        ref_logits_all = unroll_logits_eval(ref_params, rollouts)
        ref_logits_gen = ref_logits_all[:, gen_start-1:-1, :]
        
        kl = kl_from_logits(logits_gen, ref_logits_gen)
        
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
        story              = random.choice(all_stories)
        ids                = encode_text(story)
        start              = random.randint(0, max(0, len(ids) - prompt_len))
        p_ids              = np.full((prompt_len,), pad_id, dtype=np.int32)
        chunk              = ids[start:start+prompt_len]
        p_ids[:len(chunk)] = chunk
        p_list.append(p_ids)
    
    # Rollout Generation буюу Олон хувилбар үүсгэх
    prompts = np.repeat(np.stack(p_list), group_size, axis=0)
    key     = jax.random.PRNGKey(update)
    rollouts, behavior_lps = generate_rollout(grpo_state.params, jnp.asarray(prompts), key, grpo_temp)
    
    # Fluency Score буюу Reference моделиор шалгах
    # Rollouts нь [B, SeqLen], Ref model-оор оруулаад generated part-ийн logprob авна
    ref_lps_all    = unroll_logits_eval(frozen_ref, rollouts)
    # i-р индекс дээрхи logit i+1 ийг таана
    ref_gen_logits = ref_lps_all[:, prompt_len-1:-1, :]
    target_lps     = logprob_from_logits(ref_gen_logits, rollouts[:, prompt_len:])
    fluency        = np.array(jnp.mean(target_lps, axis=1))
    
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
        print(f"   >> Хамгийн сайн дээж: {best_txt[:160]}...")

print("\n=== СУРГАЛТ ДУУСЛАА ===")
# Төгсгөлийн шалгалт
final_prompt        = np.full((1, prompt_len), pad_id, dtype=np.int32)
final_prompt[0, :4] = encode_text("Once")[:4]
final_rollout, _    = generate_rollout(grpo_state.params, jnp.asarray(final_prompt), jax.random.PRNGKey(999), 0.8)
print(f"Сүүлийн үр дүн: {decode_ids(final_rollout[0, prompt_len:])}")