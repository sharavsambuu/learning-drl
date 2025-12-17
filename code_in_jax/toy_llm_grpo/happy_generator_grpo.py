#
# HAPPY TEXT GENERATOR WITH GRPO (TOY LLM HELLO WORLD) 
#
# Энэхүү код бол DeepSeek-ийн GRPO (Group Relative Policy Optimization) алгоритмын
# ажиллах зарчмыг харуулах зорилготой "Hello World" жишээ юм. 
# Моделийн үндсэн зорилго нь ямар ч эхлэл текст өгөгдсөн бай түүнийг эерэг, аз жаргалтай 
# утга агуулгатайгаар үргэлжлүүлэн бичдэг болгож сурах явдал юм. 
#
# Моделийн бүтэц:
# Модель нь 2 давхарга бүхий LSTM-ээс бүрдэх бөгөөд дээр нь Global Attention механизм 
# суурилуулсан. Attention ашигласнаар тэмдэгт бүрийг (Char-level) боловсруулахдаа өмнөх 
# бүх түүхийг найдвартайгаар санах боломжтой болдог.
#
# Машин сургалтын явц:
# 1. SFT (Supervised Fine-Tuning): TinyStories өгөгдлийн багцаас шүүсэн түүхүүд 
#    дээр суурь хэлний мэдлэг олгох.
# 2. GRPO (Alignment): Critic модель ашиглахгүйгээр, нэг prompt дээр 16 өөр хувилбар 
#    үүсгэж, тэдгээрийг хооронд нь харьцуулах замаар хамгийн сайн хариултыг дэмжих 
#    Reinforcement Learning.
#
#
# - Dynamic KL Divergence: Модель өөрийн суурь мэдлэгээсээ хэт холдох эсвэл хэт 
#   хуучинсаг байхаас сэргийлж KL-ийг динамикаар зохицуулагдана.
# - Rule-based Reward: Үгсийн олон янз байдал (diversity), текстийн бүтэц болон урт 
#   дээр суурилан Reward тооцно.
#
#

import os
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
debug_render          = True
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# SFT сургалтын тохиргоо
sft_total_steps       = 5000
sft_batch_size        = 92
sft_seq_len           = 192
sft_learning_rate     = 0.0015
sft_warmup_steps      = 400

# GRPO алгоритмын үндсэн тохиргоо
grpo_total_updates    = 1500
prompts_per_update    = 4
group_size            = 16      # Грүппийн хэмжээ
gen_len               = 192     # Үүсгэх текстийн урт
grpo_temp             = 1.0

ppo_epochs            = 3
mini_batch_size       = 92
clip_epsilon          = 0.2
entropy_coeff         = 0.02
grpo_lr               = 0.0004
max_grad_norm         = 0.5

# Dynamic KL Divergence тохиргоо
kl_beta               = 0.04    # Анхны утга
target_kl             = 0.05    # Зорилтот KL түвшин
kl_alpha              = 1.2     # KL-ийг өөрчлөх хурд

# Моделийн дотоод бүтэц
prompt_len            = 48
model_max_len         = 256
num_lstm_layers       = 2
embed_dim             = 32
hidden_dim            = 64

# Эерэг болон сөрөг үгсийн сан
happy_vocab = ["happy", "joy", "joyful", "smile", "smiled", "laugh", "laughed", "love", "loved", "kind", "nice", "fun", "good", "great", "amazing", "wonderful", "excited", "brave", "bright", "safe", "friend", "friends"]
sad_vocab   = ["sad", "cry", "cried", "bad", "angry", "mad", "hurt", "scary", "afraid", "fear", "dark", "hate", "hated", "mean", "alone", "lost", "dead", "death", "kill", "killed"]
negations   = ["not", "no", "never", "don't", "can't", "won't", "neither", "nor"]

np.random.seed(seed)
random.seed(seed)

# Өгөгдлийг уншиж цэвэрлэх хэсэг
with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

all_stories = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 0]
positive_stories = [s for s in all_stories if any(w in s.lower() for w in happy_vocab)]

# Тэмдэгтээр токенжуулах хэсэг
unique_chars   = sorted(list(set("".join(all_stories))))
PAD, BOS       = "<PAD>", "<BOS>"
chars          = [PAD, BOS] + unique_chars
char_to_id     = {c: i for i, c in enumerate(chars)}
id_to_char     = {i: c for c, i in char_to_id.items()}
vocab_size     = len(chars)
pad_id, bos_id = char_to_id[PAD], char_to_id[BOS]
fallback_id    = char_to_id.get(" ", bos_id)

def encode_text(text):
    return [bos_id] + [char_to_id.get(ch, pad_id) for ch in text]

def decode_ids(ids):
    return "".join([id_to_char[int(i)] for i in ids if int(i) not in [pad_id, bos_id]])

corpus_text = "\n\n".join(positive_stories)
corpus_ids  = np.array(encode_text(corpus_text), dtype=np.int32)

# Архитектур, LSTM + Attention
class TinyLLM(nn.Module):
    vocab_size: int; embed_dim: int; hidden_dim: int; num_layers: int; max_len: int

    @nn.compact
    def __call__(self, token_id, carry, deterministic=True):
        lstm_carries, attn_mem, step_idx = carry
        x = nn.Embed(self.vocab_size, self.embed_dim)(token_id)

        new_carries = []
        for i in range(self.num_layers):
            c_i, h_i      = lstm_carries[i]
            (c_i, h_i), h = nn.LSTMCell(self.hidden_dim, name=f"lstm_{i}")((c_i, h_i), x)
            new_carries.append((c_i, h))
            x = h

        h_expanded = x[:, None, :]
        attn_mem = jax.lax.dynamic_update_slice(attn_mem, h_expanded, (0, step_idx, 0))

        q = nn.Dense(self.hidden_dim, name="attn_q")(x)
        k = nn.Dense(self.hidden_dim, name="attn_k")(attn_mem)
        v = nn.Dense(self.hidden_dim, name="attn_v")(attn_mem)

        scores = jnp.sum(k * q[:, None, :], axis=-1) / jnp.sqrt(self.hidden_dim)
        mask   = jnp.arange(self.max_len) <= step_idx
        scores = jnp.where(mask[None, :], scores, -1e9)
        w      = jax.nn.softmax(scores, axis=-1)
        ctx    = jnp.sum(v * w[:, :, None], axis=1)

        h_mix  = nn.Dense(self.hidden_dim)(jnp.concatenate([x, ctx], axis=-1))
        h_mix  = nn.LayerNorm()(nn.tanh(h_mix))
        logits = nn.Dense(self.vocab_size)(h_mix)

        return logits, (tuple(new_carries), attn_mem, step_idx + 1)

def init_carry(batch_size):
    carries  = tuple((jnp.zeros((batch_size, hidden_dim)), jnp.zeros((batch_size, hidden_dim))) for _ in range(num_lstm_layers))
    attn_mem = jnp.zeros((batch_size, model_max_len, hidden_dim))
    return (carries, attn_mem, 0)

model  = TinyLLM(vocab_size, embed_dim, hidden_dim, num_lstm_layers, model_max_len)
params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,), jnp.int32), init_carry(1), deterministic=True)["params"]

# Туслах функцүүд: Logprob, Entropy, KL Divergence
@jax.jit
def logprob_from_logits(logits, actions):
    return jnp.take_along_axis(jax.nn.log_softmax(logits, -1), actions[:, None], 1).squeeze(1)

@jax.jit
def kl_from_logits(logits_new, logits_ref):
    p_new = jax.nn.softmax(logits_new, -1)
    return jnp.sum(p_new * (jax.nn.log_softmax(logits_new, -1) - jax.nn.log_softmax(logits_ref, -1)), -1).mean()

@jax.jit
def unroll_logits_eval(params, token_seq):
    def step_fn(carry, tok):
        logits, carry = model.apply({"params": params}, tok, carry, deterministic=True)
        return carry, logits
    _, logits_seq = jax.lax.scan(step_fn, init_carry(token_seq.shape[0]), token_seq.T)
    return jnp.transpose(logits_seq, (1, 0, 2))

@jax.jit
def generate_rollout(behavior_params, prompt_tokens, key, temperature):
    def condition_step(carry, tok):
        logits, carry = model.apply({"params": behavior_params}, tok, carry, deterministic=True)
        return carry, logits
    carry, _ = jax.lax.scan(condition_step, init_carry(prompt_tokens.shape[0]), prompt_tokens.T)

    def gen_step(carry_key_tok, _):
        carry, key, tok = carry_key_tok
        logits, carry   = model.apply({"params": behavior_params}, tok, carry, deterministic=True)
        key, subkey     = jax.random.split(key)
        next_tok        = jax.random.categorical(subkey, logits / temperature, axis=-1).astype(jnp.int32)
        return (carry, key, next_tok), (next_tok, logprob_from_logits(logits, next_tok))

    _, (gen_toks, gen_lps) = jax.lax.scan(gen_step, (carry, key, prompt_tokens[:, -1]), length=gen_len)
    return jnp.concatenate([prompt_tokens, gen_toks.T], axis=1), gen_lps.T

# Хэл шинжлэлийн дүрмүүд ба текстийн бүтэц гэх мэт дээр суурилсан Reward функц
def reward_hybrid_pro(text, fluency_score):
    t     = text.lower()
    words = re.findall(r"[a-z']+", t)
    
    # Хэт богино текстэд шийтгэл өгөх
    if len(words) < 6: return -4.0

    score         = 0.0
    happy_matches = 0
    
    # Үг бүрийг контексттэй нь шалгах
    for i, w in enumerate(words):
        if w in happy_vocab:
            # Үгүйсгэл шалгах (жишээ нь: "not happy")
            context = words[max(0, i-2):i]
            if any(n in context for n in negations):
                score -= 3.0
            else:
                score         += 2.5
                happy_matches += 1
        elif w in sad_vocab:
            score -= 2.0

    # Keyword Spamming-ээс сэргийлэх (нэг үг хэт олон давтагдах)
    for w in set(words):
        count = words.count(w)
        if count > 3: score -= (count - 3) * 1.0

    # Текстийн олон янз байдал (Unique words ratio)
    diversity = len(set(words)) / len(words)
    score    += diversity * 4.0

    # Fluency Gate: Reference моделиос хэт зөрвөл шийтгэх
    if fluency_score < -3.5: score -= 4.0

    # Төгсгөлийн цэгц болон урт (180 тэмдэгтэд ойрхон байх)
    if text.strip().endswith(('.', '!', '?')): score += 1.5
    if len(text) > 100: score += 1.0

    return max(-10.0, min(10.0, score))

# GRPO Advantage тооцоолол: Групп доторх нормчлал
def compute_grpo_advantages(rewards, n_prompts, g_size):
    rg   = rewards.reshape(n_prompts, g_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    std  = np.std (rg, axis=1, keepdims=True) + 1e-8
    adv  = (rg - mean) / std
    return adv.reshape(-1).astype(np.float32), float(mean.mean())

# SFT сургалтын хэсэг
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

print("=== PHASE 1: SFT Эхэллээ ===")
for step in range(sft_total_steps):
    starts = np.random.randint(0, corpus_ids.shape[0] - sft_seq_len - 1, sft_batch_size)
    batch = np.stack([corpus_ids[s:s+sft_seq_len+1] for s in starts])
    sft_state, sft_loss = sft_step(sft_state, jnp.asarray(batch))
    if step % 500 == 0: print(f"Алхам {step} | Loss: {sft_loss:.4f}")

# GRPO сургалтын хэсэг
grpo_state = train_state.TrainState.create(apply_fn=model.apply, params=sft_state.params, tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(grpo_lr)))
frozen_ref = grpo_state.params

@jax.jit
def grpo_step(state, ref_params, rollouts, old_lps, advs, beta):
    gen_start, gen_end = prompt_len - 1, prompt_len - 1 + gen_len
    def loss_fn(p):
        logits = unroll_logits_eval(p, rollouts[:, :-1])[:, gen_start:gen_end]
        logp   = logprob_from_logits(logits, rollouts[:, prompt_len:])
        ratio  = jnp.exp(logp - old_lps)
        surr1  = ratio * advs[:, None]
        surr2  = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advs[:, None]
        
        ref_logits = unroll_logits_eval(ref_params, rollouts[:, :-1])[:, gen_start:gen_end]
        kl         = kl_from_logits(logits.reshape((-1, vocab_size)), ref_logits.reshape((-1, vocab_size)))
        
        pg_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        return pg_loss + beta * kl, (pg_loss, kl)
    (loss, (pg, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, pg, kl

print("\n=== PHASE 2: GRPO Эхэллээ (Group Size: 16) ===")
for update in range(grpo_total_updates):
    # Prompt бэлдэх
    p_list = []
    for _ in range(prompts_per_update):
        story = random.choice(all_stories)
        ids   = encode_text(story)
        start = random.randint(0, max(0, len(ids) - prompt_len))
        p_ids = np.full((prompt_len,), pad_id, dtype=np.int32)
        chunk = ids[start:start+prompt_len]
        p_ids[:len(chunk)] = chunk
        p_list.append(p_ids)
    
    prompts = np.repeat(np.stack(p_list), group_size, axis=0)
    key     = jax.random.PRNGKey(update)
    rollouts, behavior_lps = generate_rollout(grpo_state.params, jnp.asarray(prompts), key, grpo_temp)
    
    # Fluency оноог Reference моделоос авах
    ref_lps_all = unroll_logits_eval(frozen_ref, rollouts[:, :-1])
    target_lps  = logprob_from_logits(ref_lps_all[:, prompt_len-1:-1], rollouts[:, prompt_len:])
    fluency     = np.array(jnp.mean(target_lps, axis=1))
    
    # Reward тооцох
    rewards        = np.array([reward_hybrid_pro(decode_ids(rollouts[i, prompt_len:]), fluency[i]) for i in range(rollouts.shape[0])])
    advs, m_reward = compute_grpo_advantages(rewards, prompts_per_update, group_size)
    
    # PPO Update
    for _ in range(ppo_epochs):
        grpo_state, _, pg_l, kl_l = grpo_step(grpo_state, frozen_ref, rollouts, behavior_lps, jnp.asarray(advs), kl_beta)

    # Dynamic KL Divergence зохицуулалт
    if kl_l > target_kl * 1.5:
        kl_beta *= kl_alpha
    elif kl_l < target_kl / 1.5:
        kl_beta /= kl_alpha

    if update % 50 == 0:
        print(f"Update {update:4d} | R: {m_reward:6.2f} | KL: {float(kl_l):.4f} | Beta: {kl_beta:.4f}")
        test_txt = decode_ids(rollouts[0, prompt_len:])
        print(f"Sample: {test_txt[:80]}...")

print("\n=== СУРГАЛТ ДУУСЛАА ===")
# Төгсгөлийн тест
final_prompt = np.full((1, prompt_len), pad_id, dtype=np.int32)
final_prompt[0, :4] = encode_text("Once")[:4]
final_rollout, _ = generate_rollout(grpo_state.params, jnp.asarray(final_prompt), jax.random.PRNGKey(999), 0.8)
print(f"Үр дүн: {decode_ids(final_rollout[0, prompt_len:])}")