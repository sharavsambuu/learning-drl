#
# HAPPY TEXT GENERATOR - CWRRT TRANSFORMER + GRPO TUNING
#
# CWRRT, Cross-Window Residual Recurrent Transformer, Sharavsambuu.G (2026/01/01)
#
#
# ЗОРИЛГО:
# Урт текстийг window хэсгүүдэд хувааж, өмнөх window-ийн мэдээллийг
# carry жин ашиглан дамжуулах архитектурын туршилт.
#
#
# АРХИТЕКТУР:
#   Энэ NN загвар бол урт текстийг боловсруулах зорилгоор Transformer-ийн attention
#   механизмыг recurrent санах ойтой хослуулсан эрлийз бүтэц юм. 
#   Санах ойг тусгай токен хэлбэрээр залгахын оронд Vector Injection буюу
#   вектор дээр шууд нэмэх аргыг ашигласан бөгөөд JAX-ийн хурдыг ашиглан маш хөнгөн
#   бүтцээр 1024+ урттай текстийг боловсруулах боломжтой.
#
# Cross-Window Residual Recurrent Transformer нэршил:
#
# CROSS-WINDOW (Цонх дамнасан):
#   Урт текстийг санах ойд багтаахын тулд жижиг цонхнуудад хувааж, 
#   нэг цонхноос нөгөө цонх руу мэдээлэл (overlap & memory) дамжуулах аргачлал.
#
# RESIDUAL (Үлдэгдэл буюу нэмэлт холбоос):
#   Урт хугацааны санах ойг (ssum) оролтын өгөгдөл дээр зүгээр залгах (concat) биш, 
#   харин вектор дээр шууд нэмэх (injection) буюу Residual Connection хэлбэрээр 
#   оруулж мэдээллийг гээхгүй байх.
#
# RECURRENT (Давталттай):
#   Transformer блокыг JAX-ийн lax.scan ашиглан цаг хугацааны дарааллаар, 
#   RNN (Recurrent Neural Network) шиг цувуулж ажиллуулдаг тул онолын хувьд 
#   хязгааргүй урт дарааллыг боловсруулах боломжтой.
#
# TRANSFORMER:
#   Моделийн дотоод тооцооллын үндсэн нэгж нь Self-Attention механизм дээр 
#   суурилан орчин үеийн Transformer блок.
#
#
# ХОЛБООТОЙ АЖЛУУД (REFERENCES):
#
# TRANSFORMER-XL (Google, 2019)
#   - Текстийг хэсэгчлэн унших (Segment-Level Recurrence).
#   - CWRRT дахь шийдэл, Текстийг window буюу цонхнуудад хувааж, өмнөх цонхны
#     төгсгөлийг mem дараагийн цонхны Key/Value болгон дамжуулж ашиглах.
#
# RECURRENT MEMORY TRANSFORMER - RMT (Bulatov et al., 2022)
#   - Transformer-ийг RNN шиг ажиллуулж хязгааргүй контекст үүсгэх.
#   - CWRRT дахь шийдэл, JAX lax.scan ашиглан transformer блокийг давталттайгаар
#     ажиллуулж, ssum (хураангуй) векторыг текстийн эхнээс дуустал зөөвөрлөх.
#
# COMPRESSIVE TRANSFORMER (DeepMind, 2020)
#   - Хуучин санах ойг мартахгүйн тулд шахаж хадгалах.
#   - CWRRT дахь шийдэл, Хуучин цонхны мэдээллийг хаяхын оронд ssum вектор руу
#     шахаж (compress), урт хугацааны санах ой үүсгэх.
#
# LSTM / GATED RNNS
#   - Мэдээллийн урсгалыг хаалтаар (Gate) удирдах.
#   - CWRRT дахь шийдэл, lambda болон alpha гэсэн Sigmoid gate-үүдийг ашиглаж
#     хуучин санах ой болон шинэ мэдээллийг холих харьцааг сурах.
#
#
#


import os
# JAX санах ойн тохиргоо (OOM-ээс сэргийлэх)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ]  = "platform"

import re
import gc
import math
import random
import jax
import optax
import numpy         as np
import flax.linen    as nn
from   jax           import numpy as jnp
from   flax.training import train_state


# -----------------------------
# Hyperparameters
# -----------------------------
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# CWRRT Тохиргоо (Long Context Memory)
cwr_window_len        = 256
cwr_overlap           = 64
cwr_step              = cwr_window_len - cwr_overlap
cwr_lambda_init       = 0.8
cwr_alpha_init        = 0.1

# PHASE 1: SFT
sft_total_steps       = 10000
sft_long_seq_len      = 1024
sft_batch_size        = 16
sft_learning_rate     = 3e-4
sft_warmup_steps      = 200
sft_sample_freq       = 500

# PHASE 2: GRPO
grpo_total_updates    = 5000
group_size            = 16
prompts_per_update    = 4
gen_len               = 1024
grpo_temp             = 0.9
grpo_sample_freq      = 20

# RL Тохиргоо
ppo_epochs            = 3
mini_batch_size       = 16
accum_steps           = 4
clip_epsilon          = 0.2
entropy_coeff         = 0.01
grpo_lr               = 1e-5
max_grad_norm         = 1.0

# Dynamic KL
kl_beta               = 0.04
target_kl             = 0.05
kl_alpha              = 1.2

# Моделийн дотоод бүтэц
prompt_len            = 48
num_layers            = 4
num_heads             = 8
embed_dim             = 256
max_seq_len           = 4096

# Үгсийн сан
happy_vocab = ["happy", "joy", "joyful", "smile", "smiled", "laugh", "laughed", "love", "loved", "kind", "nice", "fun", "good", "great", "amazing", "wonderful", "excited", "brave", "bright", "safe", "friend", "friends"]
sad_vocab   = ["sad", "cry", "cried", "bad", "angry", "mad", "hurt", "scary", "afraid", "fear", "dark", "hate", "hated", "mean", "alone", "lost", "dead", "death", "kill", "killed"]
negations   = ["not", "no", "never", "don't", "can't", "won't", "neither", "nor"]

np.random.seed(seed)
random.seed(seed)

# -----------------------------
# ӨГӨГДӨЛ БЭЛТГЭХ
# -----------------------------
if not os.path.exists(dataset_path):
    print("Dataset not found, using dummy text.")
    raw_text = "Once upon a time there was a happy robot. It loved to smile. " * 5000
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

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

# -----------------------------
# CWRRT МОДЕЛ (RECURRENT TRANSFORMER)
# -----------------------------

def _logit(p):
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

class CausalSelfAttention(nn.Module):
    embed_dim: int; num_heads: int
    @nn.compact
    def __call__(self, x, mask=None, kv=None, deterministic=True):
        if kv is None: kv = x
        B, Tq, C = x.shape
        _, Tk, _ = kv.shape
        head_dim = self.embed_dim // self.num_heads

        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(kv)
        v = nn.Dense(self.embed_dim)(kv)

        q = q.reshape(B, Tq, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        attn = (jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim))
        if mask is not None: attn = jnp.where(mask, attn, -1e9)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(0.1, deterministic=deterministic)(attn)

        out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)
        return nn.Dense(self.embed_dim)(out)

class MLP(nn.Module):
    embed_dim: int
    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.embed_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        return nn.Dropout(0.1, deterministic=deterministic)(x)

class Block(nn.Module):
    embed_dim: int; num_heads: int
    @nn.compact
    def __call__(self, x, mask=None, kv=None, deterministic=True):
        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(nn.LayerNorm()(x), mask, kv=kv, deterministic=deterministic)
        x = x + MLP(self.embed_dim)(nn.LayerNorm()(x), deterministic)
        return x

class CWRRTWindowCell(nn.Module):
    """
    CWRRT Cell:
    - Memory state saved before masking
    - Boolean causal mask
    - Inputs as Tuple
    """
    vocab_size   : int
    embed_dim    : int
    num_layers   : int
    num_heads    : int
    window_len   : int
    overlap      : int
    max_seq_len  : int
    lambda_init  : float
    alpha_init   : float
    deterministic: bool = True

    @nn.compact
    def __call__(self, carry, inputs):
        tokens_w, pos_offset = inputs 
        mem, ssum = carry
        
        B, T = tokens_w.shape
        O    = self.overlap

        # Embeddings
        tok_emb = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)
        pos_ids = (jnp.arange(T) + pos_offset).astype(jnp.int32)
        pos_ids = jnp.clip(pos_ids, 0, self.max_seq_len - 1)
        pos_emb = nn.Embed(self.max_seq_len, self.embed_dim)(pos_ids)
        x       = tok_emb + pos_emb[None, :, :]

        # Memory Adapter
        mem = nn.Dense(self.embed_dim, name="mem_adapter")(mem)
        mem = nn.LayerNorm(name="mem_norm")(mem)

        # Summary Injection
        alpha      = jax.nn.sigmoid(self.param("alpha_p", nn.initializers.constant(_logit(self.alpha_init)), (self.embed_dim,)))
        carry_proj = nn.Dense(self.embed_dim, use_bias=False, name="sum_proj")(ssum)
        x          = x + (carry_proj * alpha[None, :])[:, None, :]

        # Mask Preparation
        causal_tt   = jnp.tril(jnp.ones((T, T), dtype=bool))
        mem_to_all  = jnp.ones((T, O), dtype=bool)
        causal      = jnp.concatenate([mem_to_all, causal_tt], axis=1)[None, None, :, :]
        
        key_ok = jnp.concatenate(
            [jnp.ones((B, O), dtype=bool), (tokens_w != pad_id)],
            axis=1
        )
        mask = causal & key_ok[:, None, None, :]

        # Transformer Blocks (KV Loop)
        for i in range(self.num_layers):
            kv = jnp.concatenate([mem, x], axis=1) 
            x  = Block(self.embed_dim, self.num_heads, name=f"b{i}")(x, mask, kv=kv, deterministic=self.deterministic)

        x = nn.LayerNorm()(x)

        # Update Memories (masking хийхээс өмнө хадгалах)
        new_mem = x[:, -O:, :] 
        
        # Output Masking
        m = (tokens_w != pad_id).astype(jnp.float32)
        x = x * m[:, :, None] 
        
        logits = nn.Dense(self.vocab_size)(x)
        
        # Masked Pooling
        summary = jnp.sum(x * m[:, :, None], axis=1) / (jnp.sum(m, axis=1, keepdims=True) + 1e-6)
        
        lam       = jax.nn.sigmoid(self.param("lam_p", nn.initializers.constant(_logit(self.lambda_init)), (self.embed_dim,)))
        new_ssum  = (ssum * lam[None, :]) + (summary * (1.0 - lam[None, :]))

        return (new_mem, new_ssum), logits

class CWRRTTransformer(nn.Module):
    vocab_size  : int
    embed_dim   : int 
    num_layers  : int
    num_heads   : int
    window_len  : int
    overlap     : int
    max_seq_len : int
    lambda_init : float
    alpha_init  : float

    @nn.compact
    def __call__(self, tokens_long, deterministic=True):
        B, N    = tokens_long.shape
        W, O, S = self.window_len, self.overlap, self.window_len - self.overlap

        n_win      = 1 if N <= W else int(math.ceil((N - W) / S)) + 1
        total_len  = W + (n_win - 1) * S
        tokens_pad = jnp.pad(tokens_long, ((0, 0), (0, total_len - N)), constant_values=pad_id)

        starts     = (jnp.arange(n_win) * S).astype(jnp.int32)
        windows    = jax.vmap(lambda s: jax.lax.dynamic_slice(tokens_pad, (0, s), (B, W)))(starts)

        ScanCell = nn.scan(
            CWRRTWindowCell,
            variable_broadcast = "params",
            split_rngs         = {"params": False, "dropout": True},
            in_axes            = 0,
            out_axes           = 0
        )

        init_mem  = jnp.zeros((B, O, self.embed_dim))
        init_ssum = jnp.zeros((B, self.embed_dim))

        _, logits_ws = ScanCell(
            vocab_size    = self.vocab_size ,
            embed_dim     = self.embed_dim  ,
            num_layers    = self.num_layers ,
            num_heads     = self.num_heads  ,
            window_len    = self.window_len ,
            overlap       = self.overlap    ,
            max_seq_len   = self.max_seq_len,
            lambda_init   = self.lambda_init,
            alpha_init    = self.alpha_init ,
            deterministic = deterministic
        )((init_mem, init_ssum), (windows, starts))

        out = logits_ws[0]
        if n_win > 1:
            rest = logits_ws[1:, :, O:, :].transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            out  = jnp.concatenate([out, rest], axis=1)

        return out[:, :N, :]

# -----------------------------
# INITIALIZATION
# -----------------------------
model = CWRRTTransformer(
    vocab_size     , 
    embed_dim      , 
    num_layers     , 
    num_heads      ,
    cwr_window_len , 
    cwr_overlap    , 
    max_seq_len    ,
    cwr_lambda_init, 
    cwr_alpha_init
)

# -----------------------------
# JAX ТУСЛАХ ФУНКЦҮҮД
# -----------------------------
@jax.jit
def logprob_from_logits(logits, actions):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, actions[..., None], axis=-1).squeeze(-1)

@jax.jit
def kl_from_logits(logits_new, logits_ref):
    p_new = jax.nn.softmax(logits_new, -1)
    return jnp.sum(p_new * (jax.nn.log_softmax(logits_new, -1) - jax.nn.log_softmax(logits_ref, -1)), -1).mean()

@jax.jit
def generate_rollout(params, prompts, key):
    B, P = prompts.shape
    curr = jnp.pad(prompts, ((0, 0), (0, gen_len)), constant_values=pad_id)
    done = jnp.zeros((B,), dtype=jnp.bool_)

    def body(c, i):
        seq, k, done = c
        logits = model.apply({"params": params}, seq, deterministic=True)[:, P+i-1, :]
        logits = logits.at[:, pad_id].set(-1e9)
        logits = logits.at[:, bos_id].set(-1e9)
        
        k, sk  = jax.random.split(k)
        tok    = jax.random.categorical(sk, logits / grpo_temp).astype(jnp.int32)
        
        tok    = jnp.where(done, eos_id, tok)
        lp     = logprob_from_logits(logits / grpo_temp, tok)
        
        seq    = jax.lax.dynamic_update_slice(seq, tok[:, None], (0, P+i))
        done   = jnp.logical_or(done, tok == eos_id)

        return (seq, k, done), (tok, lp)

    (final, _, _), (_, lps) = jax.lax.scan(body, (curr, key, done), jnp.arange(gen_len))
    return final, lps.T

# -----------------------------
# REWARD & ADVANTAGE (RL)
# -----------------------------
def reward_hybrid_pro(text, fluency_score):
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
    rg   = rewards.reshape(n_prompts, g_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    adv  = (rg - mean)
    return np.clip(adv, -5.0, 5.0).reshape(-1).astype(np.float32), float(mean.mean())

# -----------------------------
# PHASE 1: SFT
# -----------------------------
rng    = jax.random.PRNGKey(seed)
params = model.init(rng, jnp.zeros((1, sft_long_seq_len), dtype=jnp.int32), deterministic=True)["params"]

sft_tx = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adamw(optax.warmup_cosine_decay_schedule(0, sft_learning_rate, sft_warmup_steps, sft_total_steps), weight_decay=1e-4)
)
sft_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)

@jax.jit
def sft_step(state, batch, rng):
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
print("  PHASE 1: SFT - Суурь хэлний мэдлэг (CWRRT v6 Final Polish)")
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

# SFT дууссан, санах ойг цэвэрлэх
learned_params = sft_state.params
del sft_state, sft_tx
gc.collect()

# -----------------------------
# PHASE 2: GRPO
# -----------------------------
grpo_state = train_state.TrainState.create(
    apply_fn = model.apply,
    params   = learned_params,
    tx       = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(grpo_lr))
)
frozen_ref = grpo_state.params

@jax.jit
def grpo_minibatch_update(state, ref_params, rollouts, old_lps, advs, beta):
    r_roll = rollouts.reshape(accum_steps, mini_batch_size, -1)
    r_lps  = old_lps .reshape(accum_steps, mini_batch_size, -1)
    r_adv  = advs    .reshape(accum_steps, mini_batch_size)

    def compute_grad(carry, i):
        curr_st = carry
        b_roll, b_lps, b_adv = r_roll[i], r_lps[i], r_adv[i].squeeze()

        def loss_fn(p):
            logits     = model.apply({"params": p}, b_roll, deterministic=True)[:, prompt_len-1:-1, :] / grpo_temp
            logp_act   = logprob_from_logits(logits, b_roll[:, prompt_len:])

            ratio      = jnp.exp(logp_act - b_lps)
            surr       = jnp.minimum(ratio * b_adv[:, None], jnp.clip(ratio, 1-clip_epsilon, 1+clip_epsilon) * b_adv[:, None])

            ref_logits = model.apply({"params": ref_params}, b_roll, deterministic=True)[:, prompt_len-1:-1, :] / grpo_temp
            kl         = kl_from_logits(logits, ref_logits)

            p_full     = jax.nn.softmax(logits, axis=-1)
            ent        = -jnp.sum(p_full * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean()

            return -surr.mean() + beta * kl - (entropy_coeff * ent), (kl, ent)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(curr_st.params)
        return curr_st, (grads, aux)

    _, (all_grads, auxs) = jax.lax.scan(compute_grad, state, jnp.arange(accum_steps))
    avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), all_grads)
    return state.apply_gradients(grads=avg_grads), jnp.mean(auxs[0]), jnp.mean(auxs[1])

print("\n" + "="*72)
print("  PHASE 2: GRPO - Happy бодлого сургах (CWRRT - Cross-Window Residual Recurrent Transformer)")
print(f"  Updates: {grpo_total_updates} | Group: {group_size}")
print("="*72 + "\n")

for update in range(grpo_total_updates):
    p_samples = []
    for _ in range(prompts_per_update):
        story = random.choice(all_stories)
        ids   = encode_text(story)
        start = random.randint(0, max(0, len(ids)-prompt_len))
        p_samples.append(np.pad(ids[start:start+prompt_len], (0, prompt_len), constant_values=pad_id)[:prompt_len])

    prompts = np.repeat(np.stack(p_samples), group_size, axis=0)

    rng, roll_rng = jax.random.split(rng)
    rollouts, behavior_lps = generate_rollout(grpo_state.params, jnp.asarray(prompts), roll_rng)

    ref_logits_all = model.apply({"params": frozen_ref}, rollouts, deterministic=True)[:, prompt_len-1:-1, :] / grpo_temp
    ref_lps        = logprob_from_logits(ref_logits_all, rollouts[:, prompt_len:])
    fluency        = np.array(jnp.mean(ref_lps, axis=1))

    rewards = []
    for i in range(rollouts.shape[0]):
        full_text = decode_ids(rollouts[i, prompt_len:])
        rewards.append(reward_hybrid_pro(full_text, fluency[i]))
    rewards = np.array(rewards)

    advs, m_reward = compute_grpo_advantages(rewards, prompts_per_update, group_size)

    for _ in range(ppo_epochs):
        rng, update_rng = jax.random.split(rng)
        grpo_state, kl_v, ent_v = grpo_minibatch_update(
            grpo_state, frozen_ref, rollouts, behavior_lps, jnp.asarray(advs), kl_beta
        )

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