#
#  CWRRT TRANSFORMER (SFT)
#
#  CWRRT  : Cross-Window Residual Recurrent Transformer
#
#  ЗОРИЛГО:
#   - Урт текстийг хязгаарлагдмал санах ойтой GPU дээр сургах зорилготой recurrent Transformer архитектур.
#   - Текстийг жижиг window хэсгүүдэд хувааж, өмнөх window-оос хоёр төрлийн мэдээллийг дараагийнхруу дамжуулах байдлаар ажиллана:
#       1. Memory  (mem ): Сүүлийн хэдэн токений embedding (нарийн контекст).
#       2. Summary (ssum): Window-ийн ерөнхий агуулгыг базаж шингээсэн вектор (урт хугацааны контекст).
#
#  ОНЦЛОГ:
#   - RoPE (Rotary Positional Embeddings) : Урт дараалал дээр илүү сайн ажиллахын тулд абсолют байршил биш, эргэлтийн вектор ашигласан.
#   - Gated Recurrence                    : Өмнөх мэдээллийг хэр зэрэг авахыг модель өөрөө сурна (learnable gates).
#   - Scan                                : JAX-ийн nn.scan ашиглан window хоорондын шилжилтийг хурдан тооцоолно.
#
#  АЖИЛЛУУЛАХ:
#   python sft_cwrrt_autoregressive_exp.py --steps 5000 --seq-len 1024 --batch 8 --loss-freq 10 --sample-freq 100
#
#

import os
# Санах ойн менежментийг сайжруулах тохиргоо
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

import gc
import math
import random
import argparse
import time

import numpy         as np
import jax
import jax.numpy     as jnp
import flax.linen    as nn
import optax
from   flax.training import train_state


# HYPERPARAMETERS & CONFIG 

# Өгөгдлийн тохиргоо
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# CWRRT Архитектурын тохиргоо
cwr_window_len        = 128    # Нэг удаад боловсруулах цонхны урт
cwr_overlap           = 32     # Дараагийн цонх руу дамжуулах Memory-ний урт
cwr_lambda_init       = 0.5    # Summary gate-ийн анхны утга (sigmoid-оор орно)
cwr_alpha_init        = 0.1    # Summary injection gate-ийн анхны утга

# Сургалтын (SFT) тохиргоо
sft_total_steps       = 5000
sft_long_seq_len      = 1024   # Сургах үеийн нийт дарааллын урт (Window-уудад хуваагдана)
sft_batch_size        = 8
sft_learning_rate     = 5e-4
sft_warmup_steps      = 100

# Лог болон Түүвэрлэлтийн (Sampling) тохиргоо
sft_loss_freq         = 10     # Loss хэвлэх алхамын давтамж
sft_sample_freq       = 100    # Текст үүсгэж шалгах алхамын давтамж
sample_gen_len        = 256    # Үүсгэх текстийн урт
sample_temp           = 0.8    # Sampling temperature
sample_prompt_text    = "Once upon a time there was a little robot. "

# Моделийн хэмжээ
num_layers            = 6
num_heads             = 8
embed_dim             = 512    # Head dim = 512/8 = 64
max_seq_len           = 8192

# Оптимизаци (Сургалтын тогтворжилт)
max_grad_norm         = 1.0
weight_decay          = 0.01

# Random seed тохируулах
np.random.seed(seed)
random.seed(seed)


# DATA LOADING & TOKENIZATION (Өгөгдөл бэлдэх)

print(">>> Өгөгдлийг уншиж байна...")
if not os.path.exists(dataset_path):
    print("Анхаар: Dataset олдсонгүй. Хиймэл туршилтын текст үүсгэж байна.")
    dummy_vocab = ["robot", "girl", "boy", "dog", "cat", "run", "jump", "happy", "sad", "the", "a", "is", "was"]
    raw_text    = ""
    for _ in range(2000):
        sent = " ".join(random.choices(dummy_vocab, k=10)) + ". "
        raw_text += sent + end_of_text_token
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

# Тэмдэгт түвшний (Character-level) tokenizer
all_stories    = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 10]
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

# Corpus бэлдэх (бүх текстийг тоо болгох)
flat_tokens = []
print(">>> Токенчилж байна...")
for s in all_stories:
    flat_tokens.extend(encode_text(s))
corpus_ids = np.array(flat_tokens, dtype=np.int32)

print(f"Dataset Stats: Vocab={vocab_size}, Total Tokens={len(corpus_ids)}")


# MODEL COMPONENTS (RoPE & Attention)

# Урвуу logit функц (Параметрийн анхны утга онооход хэрэглэнэ)
def _logit(p):
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

# RoPE, Rotary Positional Embeddings
def apply_rope(x, freq_cis):
    # x       : (B, T, Head, Dim)
    # freq_cis: (1, T, 1, Dim/2) - complex тоонууд
    
    # x-ийг complex тоо болгож хувиргах
    B, T, H, D = x.shape
    x_complex = jax.lax.complex(x[..., 0::2], x[..., 1::2])
    
    # Эргүүлэх (Rotation)
    x_rotated = x_complex * freq_cis
    
    # Буцаагаад бодит тоо болгох
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1).reshape(B, T, H, D)
    return x_out

# RoPE давтамж урьдчилан тооцоолох
def precompute_freqs_cis(dim, max_len, theta=10000.0):
    freqs     = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t         = jnp.arange(max_len)
    freqs     = jnp.outer(t, freqs)     # (T, Dim/2)
    freqs_cis = jnp.exp(1j * freqs)     # e^(ix) = cos(x) + i*sin(x)
    return freqs_cis[None, :, None, :]  # (1, T, 1, Dim/2)

class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        # KV нь None байвал x-ийг ашиглана (Self-Attention)
        if kv is None:
            kv = x

        B, Tq, C = x.shape
        _, Tk, _ = kv.shape
        head_dim = self.embed_dim // self.num_heads

        # Q, K, V projections
        q = nn.Dense(self.embed_dim, name="q_proj")(x)
        k = nn.Dense(self.embed_dim, name="k_proj")(kv)
        v = nn.Dense(self.embed_dim, name="v_proj")(kv)

        # Multi-head хэлбэр рүү оруулах: (B, T, Heads, HeadDim)
        q = q.reshape(B, Tq, self.num_heads, head_dim)
        k = k.reshape(B, Tk, self.num_heads, head_dim)
        v = v.reshape(B, Tk, self.num_heads, head_dim)

        # RoPE (Rotation) ашиглах
        if freqs_cis is not None:
            # Давтамжийн векторыг одоогийн уртад тааруулж таслах
            f_q = freqs_cis[:, -Tq:, :, :]
            f_k = freqs_cis[:, -Tk:, :, :]

            q = apply_rope(q, f_q)
            k = apply_rope(k, f_k)

        # Attention Scores бодох: (B, Heads, Tq, Tk)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)

        if mask is not None:
            # Mask хийх (ирээдүйг харахгүй байх)
            attn_weights = jnp.where(mask, attn_weights, -1e9)

        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(0.1, deterministic=deterministic)(attn_probs)

        # Output
        out = jnp.matmul(attn_probs, v) # (B, H, Tq, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)

        return nn.Dense(self.embed_dim, name="out_proj")(out)

class MLP(nn.Module):
    embed_dim: int
    expand_factor: int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):
        h = nn.Dense(self.embed_dim * self.expand_factor)(x)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        return nn.Dropout(0.1, deterministic=deterministic)(h)

class Block(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        # Pre-LN architecture (LayerNorm-ийг урд нь хийх)
        norm_x  = nn.LayerNorm()(x)
        norm_kv = nn.LayerNorm()(kv) if kv is not None else norm_x

        attn_out = CausalSelfAttention(self.embed_dim, self.num_heads)(
            norm_x, mask=mask, kv=norm_kv, freqs_cis=freqs_cis, deterministic=deterministic
        )
        x = x + attn_out
        x = x + MLP(self.embed_dim)(nn.LayerNorm()(x), deterministic=deterministic)
        return x


# CORE CWRRT RECURRENT CELL

class CWRRTWindowCell(nn.Module):
    vocab_size   : int
    embed_dim    : int
    num_layers   : int
    num_heads    : int
    window_len   : int
    overlap      : int
    lambda_init  : float
    alpha_init   : float
    deterministic: bool = True

    @nn.compact
    def __call__(self, carry, tokens_w):
        # carry    : (mem, ssum) -> өмнөх цонхноос ирсэн мэдээлэл
        # tokens_w : (B, W)      -> одоогийн цонхны токенууд
        mem, ssum = carry

        B, T = tokens_w.shape
        O    = self.overlap

        # Embedding
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)

        # Memory Integration (Санах ой нэгтгэл)
        # Memory-г адаптераар дамжуулж хэмжээсийг тогтворжуулна
        mem_processed = nn.Dense(self.embed_dim, name="mem_adapter")(mem)
        mem_processed = nn.LayerNorm(name="mem_norm")(mem_processed)

        # Summary Injection (Хураангуйг шингээх)
        # Summary state-ийг сурч болох alpha параметрээр жигнэн оролт дээр нэмнэ
        alpha = jax.nn.sigmoid(self.param(
            "alpha_gate",
            nn.initializers.constant(_logit(self.alpha_init)),
            (self.embed_dim,)
        ))

        # Ssum projection: (B, D) -> (B, 1, D)
        ssum_proj = nn.Dense(self.embed_dim, use_bias=False, name="ssum_proj")(ssum)
        # Оролтын бүх токенд summary-г бага зэрэг нэмж өгнө (context priming)
        x = x + (ssum_proj[:, None, :] * alpha[None, None, :])

        # Mask бэлдэх 
        # Causal mask : Одоогийн токен зөвхөн өмнөхөө болон Memory-ийг харна
        
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))            # (T, T)
        mem_mask    = jnp.ones((T, O), dtype=bool)                      # (T, O) - Memory байнга харагдана 
        full_mask   = jnp.concatenate([mem_mask, causal_mask], axis=1)  # (T, O+T)

        # Padding mask (tokens_w == pad_id үед анхаарахгүй)
        valid_curr = (tokens_w != pad_id) # (B, T)
        valid_mem  = jnp.ones((B, O), dtype=bool)
        valid_k    = jnp.concatenate([valid_mem, valid_curr], axis=1) # (B, O+T)

        # Combine causal & padding: (B, 1, T, O+T)
        mask = full_mask[None, None, :, :] & valid_k[:, None, None, :]

        # RoPE оор байршлын вектор бэлдэх
        head_dim  = self.embed_dim // self.num_heads
        freqs_cis = precompute_freqs_cis(head_dim, T + O + 10) # Хангалттай урт buffer

        # Transformer давхаргууд
        curr_x = x

        for i in range(self.num_layers):
            # KV (Key/Value) нь Memory болон Одоогийн оролтыг залгаж үүснэ
            kv_seq = jnp.concatenate([mem_processed, curr_x], axis=1)

            curr_x = Block(self.embed_dim, self.num_heads, name=f"layer_{i}")(
                curr_x, mask=mask, kv=kv_seq, freqs_cis=freqs_cis, deterministic=self.deterministic
            )

        curr_x = nn.LayerNorm()(curr_x)

        # Дараагийн Memory бэлдэх
        # Сүүлийн overlap тооны гаралтыг дараагийн memory болгож хадгална
        new_mem = curr_x[:, -O:, :]

        # Гаралтын Logit-ууд
        out_mask = (tokens_w != pad_id).astype(jnp.float32)[:, :, None]
        logits   = nn.Dense(self.vocab_size)(curr_x * out_mask)

        # Summary төлөвийг шинэчлэх
        # Window-ийн дундаж vector-ийг олно (pad tokens хасна)
        win_sum = jnp.sum(curr_x * out_mask, axis=1) / (jnp.sum(out_mask, axis=1) + 1e-6)

        # Lambda gate: Хуучин summary болон шинэ window summary-ийн харьцаа
        lam = jax.nn.sigmoid(self.param(
            "lambda_gate",
            nn.initializers.constant(_logit(self.lambda_init)),
            (self.embed_dim,)
        ))

        # Шинэчлэх дүрэм: ssum_new = lam * ssum_old + (1-lam) * current_window_summary
        new_ssum = (ssum * lam[None, :]) + (win_sum * (1.0 - lam[None, :]))

        return (new_mem, new_ssum), logits

class CWRRTTransformer(nn.Module):
    vocab_size  : int
    embed_dim   : int
    num_layers  : int
    num_heads   : int
    window_len  : int
    overlap     : int
    lambda_init : float
    alpha_init  : float

    @nn.compact
    def __call__(self, tokens_long, deterministic=True):
        # tokens_long: (B, Total_Seq_Len)
        B, N    = tokens_long.shape
        W, O, S = self.window_len, self.overlap, self.window_len - self.overlap

        # Текстийг window-уудад хуваах
        n_win = 1
        if N > W:
            n_win = int(math.ceil((N - W) / S)) + 1

        total_len_needed = W + (n_win - 1) * S
        pad_amount       = max(0, total_len_needed - N)

        tokens_pad = jnp.pad(tokens_long, ((0, 0), (0, pad_amount)), constant_values=pad_id)

        # Window эхлэх цэгүүд
        starts  = (jnp.arange(n_win) * S).astype(jnp.int32)

        # vmap ашиглан window-уудыг зүсэж авах
        windows = jax.vmap(lambda s: jax.lax.dynamic_slice(tokens_pad, (0, s), (B, W)))(starts)

        # Recurrent Scan тохиргоо (Давталт)
        ScanCell = nn.scan(
            CWRRTWindowCell,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=0, # Оролтын эхний тэнхлэгээр (windows) гүйнэ
            out_axes=0
        )

        # Анхны төлөвүүд (State)
        init_mem  = jnp.zeros((B, O, self.embed_dim))
        init_ssum = jnp.zeros((B, self.embed_dim))

        # Recurrence ажиллуулах
        _, logits_windows = ScanCell(
            vocab_size    = self.vocab_size,
            embed_dim     = self.embed_dim,
            num_layers    = self.num_layers,
            num_heads     = self.num_heads,
            window_len    = self.window_len,
            overlap       = self.overlap,
            lambda_init   = self.lambda_init,
            alpha_init    = self.alpha_init,
            deterministic = deterministic
        )((init_mem, init_ssum), windows)

        # Гаралтыг буцааж эвлүүлэх
        out = logits_windows[0] # (B, W, V)

        if n_win > 1:
            # Бусад цонхнуудын зөвхөн шинэ хэсгийг (stride) авч залгана
            rest = logits_windows[1:, :, O:, :]
            rest = rest.transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            out  = jnp.concatenate([out, rest], axis=1)

        # Анхны уртаар таслах
        return out[:, :N, :]


# Сургах болон текст үүсгэх

# Урт текст боловсруулах чадвартай моделио үүсгэх
model = CWRRTTransformer(
    vocab_size     = vocab_size,
    embed_dim      = embed_dim,
    num_layers     = num_layers,
    num_heads      = num_heads,
    window_len     = cwr_window_len,
    overlap        = cwr_overlap,
    lambda_init    = cwr_lambda_init,
    alpha_init     = cwr_alpha_init
)

@jax.jit
def train_step(state, batch, rng):
    """Нэг сургалтын алхам (Forward + Backward)"""
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(p):
        # Forward pass
        logits = model.apply(
            {"params": p},
            batch[:, :-1], # Оролт: 0..T-1
            deterministic=False,
            rngs={"dropout": dropout_rng}
        )

        # Labels: 1..T
        labels = batch[:, 1:]

        # Хэмжээ тааруулах
        logits = logits[:, :labels.shape[1], :]

        # Cross Entropy Loss (pad токенийг тооцохгүй)
        loss_t = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask   = (labels != pad_id).astype(jnp.float32)

        loss   = jnp.sum(loss_t * mask) / (jnp.sum(mask) + 1e-6)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_rng

# Хурдан текст үүсгэхэд зориулсан JIT функц
@jax.jit
def predict_step_jit(params, fixed_input):
    return model.apply({"params": params}, fixed_input, deterministic=True)

def generate(params, prompt, gen_len=100, temp=0.8):
    """Текст үүсгэх функц"""
    
    token_ids = list(encode_text(prompt))
    if token_ids[-1] == eos_id: token_ids.pop()
    
    # Generate хийх үед оролтын уртыг тогтмол байлгах хэрэгтэй
    # Ингэснээр JAX дахин дахин compile хийхгүй
    max_len = sft_long_seq_len 
    
    print(f"Текст үүсгэж байна: '{prompt}' (Max Len: {max_len})")
    
    for _ in range(gen_len):
        curr_len = len(token_ids)
        if curr_len >= max_len:
            print("Дарааллын дээд хязгаарт хүрлээ.")
            break

        # Padding хийж хэмжээг тогтмол болгох
        pad_len = max_len - curr_len
        inp_np  = np.array(token_ids + [pad_id] * pad_len, dtype=np.int32)
        inp_jax = jnp.array([inp_np]) # Shape: (1, max_len)

        # JIT функцээ дуудах (Хөрвүүлэлт хийгдэхгүй, хурдан)
        logits = predict_step_jit(params, inp_jax)
        
        # Зөвхөн сүүлийн valid токений logit-ийг авах
        next_token_logits = logits[0, curr_len - 1, :]
        
        # Sampling хийх
        next_token_logits = next_token_logits.at[pad_id].set(-1e9)
        next_token_logits = next_token_logits.at[bos_id].set(-1e9)
        
        next_token_logits = np.array(next_token_logits)
        probs             = np.exp(next_token_logits / temp)
        
        # NaN эсвэл 0-д хуваахаас сэргийлэх
        probs_sum = np.sum(probs)
        if probs_sum == 0 or np.isnan(probs_sum):
             probs = np.ones_like(probs) / len(probs)
        else:
             probs /= probs_sum
        
        next_id = np.random.choice(len(probs), p=probs)
        token_ids.append(next_id)
        
        if next_id == eos_id:
            break
            
    return decode_ids(token_ids)


# MAIN EXECUTION 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps"      , type=int  , default=sft_total_steps   )
    parser.add_argument("--seq-len"    , type=int  , default=sft_long_seq_len  )
    parser.add_argument("--batch"      , type=int  , default=sft_batch_size    )
    parser.add_argument("--loss-freq"  , type=int  , default=sft_loss_freq     )
    parser.add_argument("--sample-freq", type=int  , default=sft_sample_freq   )
    args = parser.parse_args()

    # Дэлгэцэнд мэдээлэл хэвлэх
    print("\n" + "="*60)
    print("  CWRRT TRANSFORMER - SFT EXPERIMENT")
    print(f"  Steps: {args.steps} | Batch: {args.batch} | SeqLen: {args.seq_len}")
    print(f"  Window: {cwr_window_len} | Overlap: {cwr_overlap} | Emb: {embed_dim}")
    print("="*60 + "\n")

    # Модель болон санамсаргүй тооны үүсгүүр бэлдэх
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    # Анхны утга оноох (Dummy input)
    dummy_in  = jnp.zeros((1, args.seq_len), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_in, deterministic=True)
    params    = variables["params"]

    # Параметрийн тоог тоолох
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Моделийн нийт параметр: {param_count/1e6:.2f}M")

    # Оптимизатор үүсгэх
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=1e-6,
                peak_value=sft_learning_rate,
                warmup_steps=sft_warmup_steps,
                decay_steps=args.steps,
                end_value=1e-6
            ),
            weight_decay=weight_decay
        )
    )
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Сургах давталт 
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        # Багц өгөгдөл бэлдэх (Batch)
        starts    = np.random.randint(0, len(corpus_ids) - args.seq_len - 1, args.batch)
        batch_np  = np.stack([corpus_ids[s : s + args.seq_len + 1] for s in starts])
        batch_jax = jnp.asarray(batch_np, dtype=jnp.int32)

        # Сургах алхам
        state, loss, rng = train_step(state, batch_jax, rng)

        # Лог хэвлэх
        if step % args.loss_freq == 0:
            dt = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s")
        
        # Жишээ текст үүсгэх
        if step % args.sample_freq == 0:
            print("\n" + "-"*40)
            sample_text = generate(
                state.params, 
                sample_prompt_text, 
                gen_len=sample_gen_len, 
                temp=sample_temp
            )
            print(f"ГАРСАН ҮР ДҮН: {sample_text}")
            print("-"*40 + "\n")

    print("Сургалт дууслаа!")

if __name__ == "__main__":
    main()