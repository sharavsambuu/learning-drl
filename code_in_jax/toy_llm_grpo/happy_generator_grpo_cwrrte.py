#
#  HAPPY TEXT GENERATOR - CWRRTE TRANSFORMER + GRPO TUNING
#
#  CWRRTE: A Cross-Window Recurrent Transformer with Conditional Engram Memory
#
#  ЗОРИЛГО:
#   PHASE 1: SFT  - Суурь хэлний мэдлэг (TinyStories дээр)
#   PHASE 2: GRPO - Happy чиглэлтэй бодлого сургах (Reward shaping)
#
#  АРХИТЕКТУР (CWRRTE):
#   1. Recurrent Memory (mem   ) : Өмнөх цонхны overlap embeddings
#   2. Global Summary   (ssum  ) : Урт хугацааны контекст буюу агуулгын вектор
#   3. Engram Memory    (lookup) : N-gram hash -> lookup table (KV-bank)
#
#  ЛАВЛАГАА:
#   - DeepSeek, Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
#     https://www.arxiv.org/pdf/2601.07372
#

import os
# JAX санах ойн тохиргоо (OOM-ээс сэргийлэх)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ]  = "platform"

import re
import gc
import math
import time
import random
import argparse

import jax
import optax
import numpy         as np
import flax.linen    as nn
from   jax           import numpy as jnp
from   flax.training import train_state


# HYPERPARAMETERS & CONFIGURATION

dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# CWRRTE Архитектур (Long Context)
cwr_window_len        = 256    # Цонхны урт
cwr_overlap           = 64     # Дамжуулах санах ойн хэмжээ
cwr_lambda_init       = 0.8    # Summary gate анхны утга
cwr_alpha_init        = 0.1    # Summary injection gate

# Engram Санах ой
engram_vocab_size     = 200000 # Энграм хүснэгтийн хэмжээ
engram_ngram_n        = 4      # N-gram урт
engram_dropout        = 0.05   # Санах ойгоос хэт хамаарахаас сэргийлэх

# PHASE 1, SFT (Supervised Fine-Tuning)
sft_total_steps       = 10000
sft_long_seq_len      = 1024
sft_batch_size        = 16
sft_learning_rate     = 3e-4
sft_warmup_steps      = 200
sft_sample_freq       = 500

# PHASE 2, GRPO (Group Relative Policy Optimization)
grpo_total_updates    = 5000
group_size            = 16     # Нэг prompt дээр үүсгэх хувилбарын тоо
prompts_per_update    = 4
gen_len               = 1024
grpo_temp             = 0.9
grpo_sample_freq      = 20

# RL / PPO тохиргоо
ppo_epochs            = 3
mini_batch_size       = 16
accum_steps           = 4
clip_epsilon          = 0.2
entropy_coeff         = 0.01
grpo_lr               = 1e-5
max_grad_norm         = 1.0

# Dynamic KL Penalty
kl_beta               = 0.04
target_kl             = 0.05
kl_alpha              = 1.2

# Моделийн хэмжээ
prompt_len            = 48
num_layers            = 4
num_heads             = 8
embed_dim             = 256
max_seq_len           = 4096

# Reward Vocab (эерэг ба сөрөг үгс)
happy_vocab = ["happy", "joy", "joyful", "smile", "smiled", "laugh", "laughed", "love", "loved", "kind", "nice", "fun", "good", "great", "amazing", "wonderful", "excited", "brave", "bright", "safe", "friend", "friends"]
sad_vocab   = ["sad", "cry", "cried", "bad", "angry", "mad", "hurt", "scary", "afraid", "fear", "dark", "hate", "hated", "mean", "alone", "lost", "dead", "death", "kill", "killed"]
negations   = ["not", "no", "never", "don't", "can't", "won't", "neither", "nor"]

np.random.seed(seed)
random.seed(seed)


# DATA LOADING (ӨГӨГДӨЛ БЭЛТГЭХ)

if not os.path.exists(dataset_path):
    print("Dataset олдсонгүй, хиймэл текст үүсгэж байна.")
    raw_text = "Once upon a time there was a happy robot. It loved to smile. " * 5000
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

# Тэмдэгт түвшний tokenizer
all_stories    = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 50]
unique_chars   = sorted(list(set("".join(all_stories))))

PAD, BOS, EOS  = "<PAD>", "<BOS>", "<EOS>"
chars          = [PAD, BOS, EOS] + unique_chars
char_to_id     = {c: i for i, c in enumerate(chars)}
id_to_char     = {i: c for c, i in char_to_id.items()}
vocab_size     = len(chars)
pad_id, bos_id, eos_id = char_to_id[PAD], char_to_id[BOS], char_to_id[EOS]

def encode_text(text):
    return [bos_id] + [char_to_id.get(ch, pad_id) for ch in text] + [eos_id]

def decode_ids(ids):
    return "".join([id_to_char[int(i)] for i in ids if int(i) not in [pad_id, bos_id, eos_id]])

flat = []
for s in all_stories[:2000]:
    flat.extend(encode_text(s))
corpus_ids = np.array(flat, dtype=np.int32)

print(f"Vocab Size: {vocab_size}, Tokens: {len(corpus_ids)}")


# UTILS & ROPE (ТУСЛАХ ФУНКЦУУД)

def _logit(p):
    """Магадлалыг logit руу хөрвүүлэх"""
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

def apply_rope(x, freq_cis):
    """Rotary Positional Embeddings ашиглах"""
    B, T, H, D = x.shape
    x_complex  = jax.lax.complex(x[..., 0::2], x[..., 1::2])
    x_rotated  = x_complex * freq_cis
    x_out      = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1).reshape(B, T, H, D)
    return x_out

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    """Байршлын давтамжийг урьдчилан тооцоолох"""
    freqs     = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t         = jnp.arange(max_len)
    freqs     = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis[None, :, None, :]


# ENGRAM MODULE (ENGRAM САНАХ ОЙ)

class VectorizedEngram(nn.Module):
    vocab_size   : int
    embed_dim    : int
    memory_size  : int   = 100000
    ngram_n      : int   = 4
    dropout_rate : float = 0.05

    def setup(self):
        # Static мэдлэг хадгалах embedding хүснэгт
        self.memory_table = self.param(
            "engram_table",
            nn.initializers.normal(stddev=0.02),
            (self.memory_size, self.embed_dim)
        )
        # Санах ойг ашиглах эсэхийг шийдэх gate
        self.gate_logit = self.param("engram_gate", nn.initializers.constant(-2.0), (self.embed_dim,))

        # N-gram hashing хийхэд ашиглах анхны тоонууд
        def get_primes(n):
            ps = []
            x  = 131
            for _ in range(n):
                ps.append(x)
                x = (x * 31) + 1
            return jnp.array(ps, dtype=jnp.uint32)

        self.primes = get_primes(self.ngram_n)

    @nn.compact  # Dropout ашиглахын тулд compact байх шаардлагатай
    def __call__(self, current_ids, prev_ids_overlap, deterministic=True):
        B, W = current_ids.shape
        O    = prev_ids_overlap.shape[1]

        # PAD token-уудыг 0 болгож цэвэрлэх
        current_ids      = jnp.where(current_ids      == pad_id, 0, current_ids)
        prev_ids_overlap = jnp.where(prev_ids_overlap == pad_id, 0, prev_ids_overlap)

        # Overlap урт хүрэлцээтэй эсэхийг шалгах
        assert O >= (self.ngram_n - 1)

        # Бүрэн дараалал үүсгэх
        full_seq = jnp.concatenate([prev_ids_overlap, current_ids], axis=1).astype(jnp.uint32)

        primes    = self.primes
        hash_sum  = jnp.zeros((B, W), dtype=jnp.uint32)
        start_idx = O

        # Vectorized Rolling Hash тооцоолол
        for i in range(self.ngram_n):
            s_start  = start_idx - i
            s_end    = full_seq.shape[1] - i
            chunk    = full_seq[:, s_start:s_end]
            hash_sum = hash_sum + (chunk * primes[i])

        # Хүснэгтээс хайх (Lookup)
        lookup_indices = hash_sum % self.memory_size
        retrieved_emb  = self.memory_table[lookup_indices]

        # Gate болон Dropout хэрэглэх
        gate = jax.nn.sigmoid(self.gate_logit)
        out  = retrieved_emb * gate

        out = nn.Dropout(self.dropout_rate, deterministic=deterministic)(out)
        return out


# TRANSFORMER BLOCKS (ATTENTION & LAYERS)

class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        if kv is None: kv = x

        B, Tq, C = x.shape
        _, Tk, _ = kv.shape
        head_dim = self.embed_dim // self.num_heads

        q = nn.Dense(self.embed_dim, name="q_proj")(x)
        k = nn.Dense(self.embed_dim, name="k_proj")(kv)
        v = nn.Dense(self.embed_dim, name="v_proj")(kv)

        q = q.reshape(B, Tq, self.num_heads, head_dim)
        k = k.reshape(B, Tk, self.num_heads, head_dim)
        v = v.reshape(B, Tk, self.num_heads, head_dim)

        if freqs_cis is not None:
            # RoPE-ийг эхнээс нь зөв таслаж хэрэглэх
            f_q = freqs_cis[:, :Tq, :, :]
            f_k = freqs_cis[:, :Tk, :, :]
            q   = apply_rope(q, f_q)
            k   = apply_rope(k, f_k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        if mask is not None:
            attn = jnp.where(mask, attn, -1e9)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(0.1, deterministic=deterministic)(attn)

        out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)
        return nn.Dense(self.embed_dim, name="out_proj")(out)

class MLP(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x, deterministic=True):
        h = nn.Dense(self.embed_dim * 4)(x)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        return nn.Dropout(0.1, deterministic=deterministic)(h)

class Block(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        norm_x  = nn.LayerNorm()(x)
        norm_kv = kv if kv is not None else norm_x

        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(
            norm_x, mask=mask, kv=norm_kv, freqs_cis=freqs_cis, deterministic=deterministic
        )
        x = x + MLP(self.embed_dim)(nn.LayerNorm()(x), deterministic=deterministic)
        return x


# CWRRTE MODEL (RECURRENT LOGIC)

class CWRRTEWindowCell(nn.Module):
    vocab_size        : int
    embed_dim         : int
    num_layers        : int
    num_heads         : int
    window_len        : int
    overlap           : int
    lambda_init       : float
    alpha_init        : float
    engram_vocab_size : int
    engram_ngram_n    : int
    deterministic     : bool = True

    @nn.compact
    def __call__(self, carry, tokens_w):
        mem_emb, mem_ids, ssum = carry

        B, T = tokens_w.shape
        O    = self.overlap

        # Оролтын Embedding
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)

        # Engram Retrieval (Static мэдлэг татах)
        engram_emb = VectorizedEngram(
            vocab_size   = self.vocab_size,
            embed_dim    = self.embed_dim,
            memory_size  = self.engram_vocab_size,
            ngram_n      = self.engram_ngram_n,
            dropout_rate = engram_dropout
        )(tokens_w, mem_ids, deterministic=self.deterministic)

        # Memory Adapter
        mem_processed = nn.Dense(self.embed_dim, name="mem_adapter")(mem_emb)
        mem_processed = nn.LayerNorm(name="mem_norm")(mem_processed)

        # Summary Injection
        alpha     = jax.nn.sigmoid(self.param("alpha_gate", nn.initializers.constant(_logit(self.alpha_init)), (self.embed_dim,)))
        ssum_proj = nn.Dense(self.embed_dim, use_bias=False, name="ssum_proj")(ssum)
        x         = x + (ssum_proj[:, None, :] * alpha[None, None, :])

        # KV-Bank байгуулах (Memory + Engram + Current)
        kv_seq = jnp.concatenate([mem_processed, engram_emb, x], axis=1)

        # Masking Strategy
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        mem_mask    = jnp.ones((T, O), dtype=bool)
        engram_mask = causal_mask
        full_mask   = jnp.concatenate([mem_mask, engram_mask, causal_mask], axis=1)

        valid_curr  = (tokens_w != pad_id)
        valid_mem   = jnp.ones((B, O), dtype=bool)
        valid_eng   = valid_curr
        valid_k     = jnp.concatenate([valid_mem, valid_eng, valid_curr], axis=1)

        mask = full_mask[None, None, :, :] & valid_k[:, None, None, :]

        # RoPE давтамж
        total_kv_len = O + T + T
        freqs_cis    = precompute_freqs_cis(self.embed_dim // self.num_heads, total_kv_len + 32)

        # Transformer Давхаргууд
        curr_x = x
        for i in range(self.num_layers):
            curr_x = Block(self.embed_dim, self.num_heads, name=f"b{i}")(
                curr_x, mask=mask, kv=kv_seq, freqs_cis=freqs_cis, deterministic=self.deterministic
            )

        curr_x = nn.LayerNorm()(curr_x)

        # Дараагийн төлөвийг бэлдэх
        new_mem_emb = curr_x[:, -O:, :]
        new_mem_ids = tokens_w[:, -O:]

        # Summary шинэчлэх
        out_mask = (tokens_w != pad_id).astype(jnp.float32)[:, :, None]
        win_sum  = jnp.sum(curr_x * out_mask, axis=1) / (jnp.sum(out_mask, axis=1) + 1e-6)

        lam      = jax.nn.sigmoid(self.param("lambda_gate", nn.initializers.constant(_logit(self.lambda_init)), (self.embed_dim,)))
        new_ssum = (ssum * lam[None, :]) + (win_sum * (1.0 - lam[None, :]))

        logits = nn.Dense(self.vocab_size)(curr_x)
        return (new_mem_emb, new_mem_ids, new_ssum), logits

class CWRRTETransformer(nn.Module):
    vocab_size        : int
    embed_dim         : int
    num_layers        : int
    num_heads         : int
    window_len        : int
    overlap           : int
    lambda_init       : float
    alpha_init        : float
    engram_vocab_size : int
    engram_ngram_n    : int

    @nn.compact
    def __call__(self, tokens_long, deterministic=True):
        B, N    = tokens_long.shape
        W, O, S = self.window_len, self.overlap, self.window_len - self.overlap

        # Текстийг цонхнуудад хуваах
        n_win      = 1 if N <= W else int(math.ceil((N - W) / S)) + 1
        total_len  = W + (n_win - 1) * S
        tokens_pad = jnp.pad(tokens_long, ((0, 0), (0, total_len - N)), constant_values=pad_id)

        starts  = (jnp.arange(n_win) * S).astype(jnp.int32)
        windows = jax.vmap(lambda s: jax.lax.dynamic_slice(tokens_pad, (0, s), (B, W)))(starts)

        # Scan ашиглан recurrent гүйцэтгэл хийх
        ScanCell = nn.scan(
            CWRRTEWindowCell,
            variable_broadcast = "params",
            split_rngs         = {"params": False, "dropout": True},
            in_axes            = 0,
            out_axes           = 0
        )

        init_mem_emb = jnp.zeros((B, O, self.embed_dim))
        init_mem_ids = jnp.zeros((B, O), dtype=jnp.int32)
        init_ssum    = jnp.zeros((B, self.embed_dim))

        _, logits_ws = ScanCell(
            vocab_size        = self.vocab_size,
            embed_dim         = self.embed_dim,
            num_layers        = self.num_layers,
            num_heads         = self.num_heads,
            window_len        = self.window_len,
            overlap           = self.overlap,
            lambda_init       = self.lambda_init,
            alpha_init        = self.alpha_init,
            engram_vocab_size = self.engram_vocab_size,
            engram_ngram_n    = self.engram_ngram_n,
            deterministic     = deterministic
        )((init_mem_emb, init_mem_ids, init_ssum), windows)

        # Гаралтыг эвлүүлэх
        out = logits_ws[0]
        if n_win > 1:
            rest = logits_ws[1:, :, O:, :].transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            out  = jnp.concatenate([out, rest], axis=1)

        return out[:, :N, :]


# МОДЕЛЬ ҮҮСГЭХ

model = CWRRTETransformer(
    vocab_size        = vocab_size,
    embed_dim         = embed_dim,
    num_layers        = num_layers,
    num_heads         = num_heads,
    window_len        = cwr_window_len,
    overlap           = cwr_overlap,
    lambda_init       = cwr_lambda_init,
    alpha_init        = cwr_alpha_init,
    engram_vocab_size = engram_vocab_size,
    engram_ngram_n    = engram_ngram_n
)


# JAX HELPERS (МАШИН СУРГАЛТЫН ТУСЛАХ ФУНКЦУУД)

@jax.jit
def logprob_from_logits(logits, actions):
    """Logits-оос сонгогдсон үйлдлийн log probability-г олох"""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, actions[..., None], axis=-1).squeeze(-1)

@jax.jit
def kl_from_logits(logits_new, logits_ref):
    """Хоёр моделийн гаралтын KL Divergence тооцох"""
    p_new = jax.nn.softmax(logits_new, -1)
    return jnp.sum(p_new * (jax.nn.log_softmax(logits_new, -1) - jax.nn.log_softmax(logits_ref, -1)), -1).mean()

@jax.jit
def generate_rollout(params, prompts, key):
    """Өгөгдсөн prompt дээр текст үүсгэх (Rollout)"""
    B, P = prompts.shape
    curr = jnp.pad(prompts, ((0, 0), (0, gen_len)), constant_values=pad_id)
    done = jnp.zeros((B,), dtype=jnp.bool_)

    def body(c, i):
        seq, k, done = c

        # Scan дотор model.apply дуудах (бүтэн дарааллаар нь)
        # Recurrent модель учраас stateful биш, бүтэн оролт өгөх шаардлагатай
        logits = model.apply({"params": params}, seq, deterministic=True)[:, P+i-1, :]
        logits = logits.at[:, pad_id].set(-1e9)
        logits = logits.at[:, bos_id].set(-1e9)

        k, sk = jax.random.split(k)
        tok   = jax.random.categorical(sk, logits / grpo_temp).astype(jnp.int32)

        tok   = jnp.where(done, eos_id, tok)
        lp    = logprob_from_logits(logits / grpo_temp, tok)

        seq   = jax.lax.dynamic_update_slice(seq, tok[:, None], (0, P+i))
        done  = jnp.logical_or(done, tok == eos_id)

        return (seq, k, done), (tok, lp)

    (final, _, _), (_, lps) = jax.lax.scan(body, (curr, key, done), jnp.arange(gen_len))
    return final, lps.T


# REWARD & ADVANTAGE (ШАГНАЛ БОЛОН ADVANTAGE)

def reward_hybrid_pro(text, fluency_score):
    """Текстийн "Happy" шинж чанарыг үнэлэх функц"""
    t     = text.lower()
    words = re.findall(r"[a-z']+", t)
    if len(words) < 6: return -5.0

    trigrams = set()
    for i in range(len(words) - 2):
        tg = (words[i], words[i+1], words[i+2])
        if tg in trigrams: return -5.0
        trigrams.add(tg)

    score = 0.0
    for i, w in enumerate(words):
        if w in happy_vocab:
            context = words[max(0, i-2):i]
            score += (-3.0 if any(n in context for n in negations) else 2.5)
        elif w in sad_vocab:
            score -= 2.0

    diversity = len(set(words)) / len(words)
    score += diversity * 4.0
    if fluency_score < -3.5: score -= 3.0

    return float(np.clip(score, -10, 10))

def compute_grpo_advantages(rewards, n_prompts, g_size):
    """Group Relative Policy Optimization - Advantage тооцох"""
    rg   = rewards.reshape(n_prompts, g_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    # Бүлэг доторх дундажтай харьцуулж advantage гаргах
    adv  = (rg - mean)
    return np.clip(adv, -5.0, 5.0).reshape(-1).astype(np.float32), float(mean.mean())


# PHASE 1, SFT (SUPERVISED FINE-TUNING)

rng    = jax.random.PRNGKey(seed)
params = model.init(rng, jnp.zeros((1, sft_long_seq_len), dtype=jnp.int32), deterministic=True)["params"]

sft_tx = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adamw(
        optax.warmup_cosine_decay_schedule(0.0, sft_learning_rate, sft_warmup_steps, sft_total_steps),
        weight_decay=1e-4
    )
)
sft_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)

@jax.jit
def sft_step(state, batch, rng):
    """SFT сургалтын алхам"""
    def loss_fn(p):
        logits = model.apply({"params": p}, batch[:, :-1], deterministic=False, rngs={'dropout': rng})
        labels = batch[:, 1:]
        logits = logits[:, :labels.shape[1], :]
        mask   = (labels != pad_id).astype(jnp.float32)
        loss   = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.sum(loss * mask) / (jnp.sum(mask) + 1e-6)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

print("\n" + "="*72)
print("  PHASE 1: SFT - Суурь хэлний мэдлэг (CWRRTE)")
print(f"  Steps: {sft_total_steps} | Batch: {sft_batch_size} | Dim: {embed_dim}")
print("="*72 + "\n")

for step in range(sft_total_steps):
    starts = np.random.randint(0, corpus_ids.shape[0] - (sft_long_seq_len + 1), sft_batch_size)
    batch  = np.stack([corpus_ids[s:s+sft_long_seq_len+1] for s in starts])

    rng, step_rng   = jax.random.split(rng)
    sft_state, loss = sft_step(sft_state, jnp.asarray(batch), step_rng)

    if step % 500 == 0:
        print(f"[SFT] Алхам {step:5d} | Loss: {float(loss):.4f}")

    if step > 0 and step % sft_sample_freq == 0:
        test_key    = jax.random.PRNGKey(step)
        test_prompt = jnp.array([encode_text("Once upon a time")[:prompt_len]])
        sample, _   = generate_rollout(sft_state.params, test_prompt, test_key)
        print("-" * 50)
        print(f"   >> Дээж (Step {step}):")
        print(decode_ids(sample[0]))
        print("-" * 50)

learned_params = sft_state.params
del sft_state, sft_tx
gc.collect()


# PHASE 2, GRPO (GROUP RELATIVE POLICY OPTIMIZATION)

grpo_state = train_state.TrainState.create(
    apply_fn = model.apply,
    params   = learned_params,
    tx       = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(grpo_lr))
)
frozen_ref = grpo_state.params

@jax.jit
def grpo_minibatch_update(state, ref_params, rollouts, old_lps, advs, beta):
    """GRPO Update (PPO-style clip loss with KL penalty)"""
    r_roll = rollouts.reshape(accum_steps, mini_batch_size, -1)
    r_lps  = old_lps .reshape(accum_steps, mini_batch_size, -1)
    r_adv  = advs    .reshape(accum_steps, mini_batch_size)

    def compute_grad(carry, i):
        curr_st = carry
        b_roll, b_lps, b_adv = r_roll[i], r_lps[i], r_adv[i].squeeze()

        def loss_fn(p):
            logits   = model.apply({"params": p}, b_roll, deterministic=True)[:, prompt_len-1:-1, :] / grpo_temp
            logp_act = logprob_from_logits(logits, b_roll[:, prompt_len:])

            ratio = jnp.exp(logp_act - b_lps)
            surr  = jnp.minimum(
                ratio * b_adv[:, None],
                jnp.clip(ratio, 1-clip_epsilon, 1+clip_epsilon) * b_adv[:, None]
            )

            ref_logits = model.apply({"params": ref_params}, b_roll, deterministic=True)[:, prompt_len-1:-1, :] / grpo_temp
            kl         = kl_from_logits(logits, ref_logits)

            p_full = jax.nn.softmax(logits, axis=-1)
            ent    = -jnp.sum(p_full * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean()

            return -surr.mean() + beta * kl - (entropy_coeff * ent), (kl, ent)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(curr_st.params)
        return curr_st, (grads, aux)

    _, (all_grads, auxs) = jax.lax.scan(compute_grad, state, jnp.arange(accum_steps))
    avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), all_grads)
    return state.apply_gradients(grads=avg_grads), jnp.mean(auxs[0]), jnp.mean(auxs[1])

print("\n" + "="*72)
print("  PHASE 2: GRPO - Happy бодлого сургах (CWRRTE)")
print(f"  Updates: {grpo_total_updates} | Group: {group_size}")
print("="*72 + "\n")

for update in range(grpo_total_updates):
    # Prompt сонгох
    p_samples = []
    for _ in range(prompts_per_update):
        story = random.choice(all_stories)
        ids   = encode_text(story)
        start = random.randint(0, max(0, len(ids)-prompt_len))
        p_samples.append(np.pad(ids[start:start+prompt_len], (0, prompt_len), constant_values=pad_id)[:prompt_len])

    prompts = np.repeat(np.stack(p_samples), group_size, axis=0)

    # Rollout хийх
    rng, roll_rng = jax.random.split(rng)
    rollouts, behavior_lps = generate_rollout(grpo_state.params, jnp.asarray(prompts), roll_rng)

    # Reference Logits & Rewards тооцох
    ref_logits_all = model.apply({"params": frozen_ref}, rollouts, deterministic=True)[:, prompt_len-1:-1, :] / grpo_temp
    ref_lps        = logprob_from_logits(ref_logits_all, rollouts[:, prompt_len:])
    fluency        = np.array(jnp.mean(ref_lps, axis=1))

    rewards = []
    for i in range(rollouts.shape[0]):
        full_text = decode_ids(rollouts[i, prompt_len:])
        rewards.append(reward_hybrid_pro(full_text, fluency[i]))
    rewards = np.array(rewards)

    # Advantages тооцох
    advs, m_reward = compute_grpo_advantages(rewards, prompts_per_update, group_size)

    # GRPO Update
    for _ in range(ppo_epochs):
        grpo_state, kl_v, ent_v = grpo_minibatch_update(
            grpo_state, frozen_ref, rollouts, behavior_lps, jnp.asarray(advs), kl_beta
        )

    # Dynamic KL adjustment
    kl_val = float(kl_v)
    if kl_val > target_kl * 1.5: kl_beta *= kl_alpha
    elif kl_val < target_kl / 1.5: kl_beta /= kl_alpha

    if update % 20 == 0:
        print(f"[GRPO] Upd {update:4d} | AvgReward: {m_reward:6.2f} | KL: {kl_val:.4f} | Beta: {kl_beta:.4f}")

    if update % grpo_sample_freq == 0:
        best_idx = np.argmax(rewards)
        print("-" * 50)
        print(f"   >> Шилдэг дээж (Update {update}):")
        print(decode_ids(rollouts[best_idx]))
        print("-" * 50)

print("\n=== СУРГАЛТ ДУУСЛАА ===")