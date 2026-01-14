
#
#  CWRRTE : A Cross-Window Recurrent Transformer with Multi-Head Engram Memory
#
#  АРХИТЕКТУРЫН ТАЙЛБАР:
#   Энэ модель нь Recurrent болон Retrieval (Хайлт) аргуудыг хослуулсан
#   архитектурын туршилт юм. Дараах үндсэн гурван бүрэлдэхүүн хэсэгтэй:
#
#   1. Recurrent Memory (mem):
#      Өмнөх цонхны сүүлийн хэсгийг (overlap) дараагийн цонх руу дамжуулж,
#      дарааллын тасралтгүй байдлыг хангана.
#
#   2. Global Summary (ssum):
#      Текстийн ерөнхий агуулгыг нэг вектор руу шахаж, урт хугацааны контекст
#      хадгалах үүрэгтэй. Gated Update ашиглана.
#
#   3. Multi-Head Engram Memory:
#      DeepSeek аргачлалын олон толгойтой (Multi-Head) хувилбар.
#      - Нэг толгойтой (Single-Head) үед үүсдэг хэш мөргөлдөөнийг арилгахын тулд
#        4-8 өөр төрлийн анхны тоо ашиглан зэрэгцүүлэн хайлт хийнэ.
#      - JAX vmap ашиглан толгой бүрийн хайлтыг GPU дээр зэрэг гүйцэтгэнэ.
#
#
#  Сайжруулалтууд болон бүрэлдэхүүн хэсгүүд:
#   - RMSNorm                 : LayerNorm-оос хөнгөн бөгөөд Deep NN training тогтворжуулна.
#   - SwiGLU                  : Google PaLM, Llama-3 загваруудын ашигладаг Gated MLP хувилбар. Логик сэтгэлгээг сайжруулна гэлцдэг.
#   - Vectorized Rolling Hash : N-gram хэшийг Python давталтгүйгээр, GPU дээр зэрэг бодно.
#   - KV-Bank Integration     : [Memory | Engram | Current] дарааллаар Attention-д орно.
#   - RoPE                    : Байршлын векторыг дарааллын эхнээс зөв хуваарилна.
#   - JAX Scan                : Цонх хоорондын рекуррент шилжилтийг өндөр хурдаар гүйцэтгэнэ.
#
#
#  TTS eSpeak суулгах (текст сонсох):
#   - Ubuntu/WSL:  sudo apt install espeak-ng
#
#
#  Лавлагаа:
#   - DeepSeek, Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs
#     https://www.arxiv.org/pdf/2601.07372
#   - GLU Variants Improve Transformer (SwiGLU paper)
#     https://arxiv.org/abs/2002.05202
#
#


import os
# Санах ойн хуваарилалтыг оновчтой болгох
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

import gc
import math
import random
import argparse
import time
import re
import shutil
import subprocess

import numpy         as np
import jax
import jax.numpy     as jnp
import flax.linen    as nn
import optax
from   flax.training import train_state


# ТОХИРГОО БОЛОН HYPERPARAMETERS

# Өгөгдөл
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# CWRRTE Архитектур
cwr_window_len        = 128    # Нэг удаад боловсруулах цонхны урт
cwr_overlap           = 32     # Дараагийн цонх руу дамжуулах overlap урт
cwr_lambda_init       = 0.5    # Summary шинэчлэх gate-ийн анхны утга
cwr_alpha_init        = 0.1    # Summary ашиглах gate-ийн анхны утга

# Engram Санах ой (Multi-Head)
engram_vocab_size     = 100000 # Энграм хүснэгтийн нийт мөр (Slot count)
engram_ngram_n        = 4      # Хэдэн тэмдэгт харж хэш хийх вэ (N-gram)
engram_num_heads      = 4      # Санах ойг хэдэн "толгой"-гоор зэрэг хайх вэ
engram_dropout        = 0.05   # Санах ойгоос хэт хамаарахаас сэргийлэх dropout

# Supervised Fine Tuning (SFT)
sft_total_steps       = 5000
sft_long_seq_len      = 1024   # Сургах дарааллын нийт урт
sft_batch_size        = 8
sft_learning_rate     = 5e-4
sft_warmup_steps      = 100

# Log болон туршилт
sft_loss_freq         = 10
sft_sample_freq       = 100
sample_gen_len        = 256
sample_temp           = 0.8
sample_prompt_text    = "Once upon a time there was a little robot. "

# eSpeak TTS
tts_enabled           = True   
tts_voice             = "en"   
tts_speed             = 165    # 80-450 орчим
tts_amp               = 120    # 0-200
tts_max_chars         = 400    # Хэт урт текст хэлэхээс сэргийлнэ

# Моделийн хэмжээ
num_layers            = 6
num_heads             = 8
embed_dim             = 512
max_seq_len           = 8192

# Оновчлол
max_grad_norm         = 1.0
weight_decay          = 0.01

# Random seed тохиргоо
np.random.seed(seed)
random.seed(seed)


# ESPEAK NON-BLOCKING TTS (Давталт саатуулахгүй)

_ESPEAK_BIN = shutil.which("espeak-ng") or shutil.which("espeak")

def _tts_clean(s, max_chars=400):
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = "".join(ch for ch in s if ch.isprintable())
    return s[:max_chars]

def speak_async(text, voice="en", speed=165, amp=120, enabled=True):
    if (not enabled) or (not _ESPEAK_BIN):
        return

    t = _tts_clean(text, max_chars=tts_max_chars)
    if not t:
        return

    try:
        p = subprocess.Popen(
            [_ESPEAK_BIN, "-v", voice, "-s", str(speed), "-a", str(amp), "--stdin"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        p.stdin.write((t + "\n").encode("utf-8", errors="ignore"))
        p.stdin.close()
    except Exception:
        pass


# ӨГӨГДӨЛ БЭЛТГЭХ (DATA LOADING)

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

# Тэмдэгт түвшний (Character-level) tokenizer бэлдэх
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

# Бүх текстийг токенчилж нэг урт массив болгох
flat_tokens = []
print(">>> Токенчилж байна...")
for s in all_stories:
    flat_tokens.extend(encode_text(s))
corpus_ids = np.array(flat_tokens, dtype=np.int32)

print(f"Dataset Stats: Vocab={vocab_size}, Total Tokens={len(corpus_ids)}")


# ТУСЛАХ ФУНКЦУУД (ROPE, UTILS)

def _logit(p):
    """Магадлалыг logit руу хөрвүүлэх (Parameter initialization)"""
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

def apply_rope(x, freq_cis):
    """
    Rotary Positional Embeddings (RoPE) ашиглан байршлын мэдээлэл оруулах.
    x: (B, T, H, D)
    freq_cis: (1, T, 1, D/2)
    """
    B, T, H, D = x.shape
    x_complex = jax.lax.complex(x[..., 0::2], x[..., 1::2])
    # Эргүүлэх үйлдэл (Rotation)
    x_rotated = x_complex * freq_cis
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1).reshape(B, T, H, D)
    return x_out

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    """RoPE давтамжийн матрицыг урьдчилан тооцоолох"""
    freqs     = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t         = jnp.arange(max_len)
    freqs     = jnp.outer(t, freqs)     # (T, Dim/2)
    freqs_cis = jnp.exp(1j * freqs)     # e^(ix)
    return freqs_cis[None, :, None, :]


# RMSNorm, SwiGLU

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    LayerNorm-ийг бодвол mean (дундаж) тооцохгүй тул хурдан, 
    мөн Deep NN-д gradient урсгалыг тогтвортой байлгана.
    """
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        # Mean хасахгүй, зөвхөн RMS-ээр хуваана
        rms   = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * scale

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    GELU-г орлоно. Gate болон Value салаатай тул логик сэтгэлгээг сайжруулдаг. 
    Llama-3, DeepSeek зэрэг орчин үеийн загваруудын стандарт.
    """
    embed_dim     : int
    expand_factor : int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden_dim = int(self.embed_dim * self.expand_factor)

        # Gate болон Value-г нэг дор үүсгээд split хийх нь JAX-д хурдан
        gate_val = nn.Dense(hidden_dim * 2, use_bias=False)(x)
        gate, val = jnp.split(gate_val, 2, axis=-1)

        # Swish Activation: x * sigmoid(x)
        act = nn.silu(gate) * val
        out = nn.Dense(self.embed_dim, use_bias=False)(act)

        return nn.Dropout(0.1, deterministic=deterministic)(out)


# MULTI-HEAD ENGRAM MEMORY

class MultiHeadEngram(nn.Module):
    """
    Олон толгойтой Engram санах ой.
    Хуучин single-head hashing нь мөргөлдөөн (collision) ихтэй байх боломжтой.
    Сайжруулалт:
      1. Head бүр өөр анхны тоонууд ашиглаж зэрэгцээ хайлт хийнэ.
      2. JAX vmap ашиглан толгойнуудыг vectorization хийнэ.
      3. Үр дүнг нэгтгэхдээ Gating ашиглана.
    """
    vocab_size   : int
    embed_dim    : int
    memory_size  : int   = 100000   # Санах ойн хүснэгтийн хэмжээ
    ngram_n      : int   = 4        # N-gram урт
    num_heads    : int   = 4        # Толгойн тоо
    dropout_rate : float = 0.05

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, "Embed dim must be divisible by heads"
        self.head_dim = self.embed_dim // self.num_heads

        # Санах ойн хүснэгт, (Memory_Size, Num_Heads, Head_Dim)
        self.memory_table = self.param(
            "engram_table",
            nn.initializers.normal(stddev=0.02),
            (self.memory_size, self.num_heads, self.head_dim)
        )

        # Gating механизм, толгой тус бүрт gate байна
        self.gate_logit = self.param(
            "engram_gate",
            nn.initializers.constant(-2.0),
            (self.num_heads, self.head_dim)
        )

        # Толгой бүрт ялгаатай анхны тоонууд үүсгэх (Deterministic)
        ps   = []
        base = 131
        for h in range(self.num_heads):
            x   = base + h * 1009
            row = []
            for _ in range(self.ngram_n):
                row.append(x)
                x = (x * 31) + 1
            ps.append(row)
        self.primes = jnp.array(ps, dtype=jnp.uint32)  # (H, N)

    @nn.compact
    def __call__(self, current_ids, prev_ids_overlap, deterministic=True):
        """
        current_ids      : (B, W)
        prev_ids_overlap : (B, O)
        """
        B, W = current_ids.shape
        O    = prev_ids_overlap.shape[1]

        # Overlap нь N-gram тооцоход хүрэлцэхүйц байх ёстой
        assert O >= (self.ngram_n - 1)

        # PAD токенууд хэшлэлтийг эвдэхээс сэргийлэх (Collision багасгана)
        current_ids      = jnp.where(current_ids      == pad_id, 0, current_ids)
        prev_ids_overlap = jnp.where(prev_ids_overlap == pad_id, 0, prev_ids_overlap)

        # Бүрэн контекст үүсгэх
        full_seq  = jnp.concatenate([prev_ids_overlap, current_ids], axis=1).astype(jnp.uint32)
        start_idx = O

        # Multi-Head Vectorized Rolling Hash
        # Гаралт: (B, W, H)
        hash_sums = jnp.zeros((B, W, self.num_heads), dtype=jnp.uint32)

        for i in range(self.ngram_n):
            s_start  = start_idx - i
            s_end    = full_seq.shape[1] - i
            chunk    = full_seq[:, s_start:s_end]                              # (B, W)
            p_vec    = self.primes[:, i]                                       # (H,)
            
            # Broadcasting: (B, W, 1) * (1, 1, H) -> (B, W, H)
            hash_sums = hash_sums + (chunk[:, :, None] * p_vec[None, None, :])

        lookup_indices = hash_sums % self.memory_size                          # (B, W, H)

        # Head-wise gather (Толгой бүр өөрийн баганаас татах)
        # table_h : (H, M, Dh)
        # idx_h   : (H, B, W)
        table_h = jnp.transpose(self.memory_table, (1, 0, 2))
        idx_h   = jnp.transpose(lookup_indices, (2, 0, 1))

        # vmap ашиглан Head dimension дээр зэрэгцүүлж хайна
        def _gather(tbl, idx):
            return tbl[idx]                                                    # (B, W, Dh)

        got_h = jax.vmap(_gather, in_axes=(0, 0), out_axes=0)(table_h, idx_h)  # (H, B, W, Dh)
        retrieved = jnp.transpose(got_h, (1, 2, 0, 3))                         # (B, W, H, Dh)

        # Gating болон нэгтгэл
        gate = jax.nn.sigmoid(self.gate_logit)                                 # (H, Dh)
        out  = retrieved * gate[None, None, :, :]                              # (B, W, H, Dh)

        out  = out.reshape(B, W, self.embed_dim)                               # (B, W, D)
        out  = nn.Dropout(self.dropout_rate, deterministic=deterministic)(out)
        return out


# TRANSFORMER BLOCKS (Pre-RMSNorm & SwiGLU)

class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        # Хэрэв kv өгөгдөөгүй бол өөрөө өөртөө attend хийнэ
        if kv is None: kv = x

        B, Tq, _ = x.shape
        _, Tk, _ = kv.shape     # KV нь (Mem + Engram + Current) учир урт байна
        head_dim = self.embed_dim // self.num_heads

        # Q, K, V 
        q = nn.Dense(self.embed_dim, name="q_proj")(x)
        k = nn.Dense(self.embed_dim, name="k_proj")(kv)
        v = nn.Dense(self.embed_dim, name="v_proj")(kv)

        # Multi-head хэлбэрт оруулах
        q = q.reshape(B, Tq, self.num_heads, head_dim)
        k = k.reshape(B, Tk, self.num_heads, head_dim)
        v = v.reshape(B, Tk, self.num_heads, head_dim)

        # RoPE, Байршлын мэдээлэл нэмэх
        if freqs_cis is not None:
            f_q = freqs_cis[:, :Tq, :, :]
            f_k = freqs_cis[:, :Tk, :, :]
            q = apply_rope(q, f_q)
            k = apply_rope(k, f_k)

        # Attention Scores тооцоолох
        q = q.transpose(0, 2, 1, 3) # (B, H, Tq, D)
        k = k.transpose(0, 2, 1, 3) # (B, H, Tk, D)
        v = v.transpose(0, 2, 1, 3) # (B, H, Tk, D)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)

        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(0.1, deterministic=deterministic)(attn_probs)

        # Гаралт
        out = jnp.matmul(attn_probs, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)

        return nn.Dense(self.embed_dim, name="out_proj")(out)

class TransformerBlock(nn.Module):
    """
    Pre-RMSNorm бүтэцтэй. MLP нь SwiGLU-тэй.
    """
    embed_dim : int
    num_heads : int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        # Attention хэсэг (Pre-RMSNorm)
        norm_x  = RMSNorm(self.embed_dim)(x)
        norm_kv = kv if kv is not None else norm_x

        attn_out = CausalSelfAttention(self.embed_dim, self.num_heads)(
            norm_x, mask=mask, kv=norm_kv, freqs_cis=freqs_cis, deterministic=deterministic
        )
        x = x + attn_out

        # MLP хэсэг (Pre-RMSNorm + SwiGLU)
        norm_x2 = RMSNorm(self.embed_dim)(x)
        mlp_out = SwiGLU(self.embed_dim)(norm_x2, deterministic=deterministic)
        x = x + mlp_out

        return x


# CWRRTE RECURRENT CELL

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
    engram_num_heads  : int          # Heads param
    deterministic     : bool = True

    @nn.compact
    def __call__(self, carry, tokens_w):
        # Carry задлах
        # mem_emb : Recurrent memory (Context)
        # mem_ids : Engram хэшлэлд зориулсан ID
        # ssum    : Global Summary вектор
        mem_emb, mem_ids, ssum = carry

        B, T = tokens_w.shape
        O    = self.overlap

        # Одоогийн оролтын Embedding
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)

        # Engram Retrieval (Multi-Head)
        # Engram нь KV-Bank-д нэмэгдэх тул түүнийг тусад нь боловсруулна
        engram_emb = MultiHeadEngram(
            vocab_size   = self.vocab_size,
            embed_dim    = self.embed_dim,
            memory_size  = self.engram_vocab_size,
            ngram_n      = self.engram_ngram_n,
            num_heads    = self.engram_num_heads,
            dropout_rate = engram_dropout
        )(tokens_w, mem_ids, deterministic=self.deterministic)

        # Retrieval вектор болон Model embedding хоёрын далайц (scale) 
        # зөрөхөөс сэргийлж Engram-ийг RMSNorm-оор тэнцвэржүүлнэ.
        engram_emb = RMSNorm(self.embed_dim)(engram_emb)

        # Memory боловсруулах
        mem_processed = nn.Dense(self.embed_dim, name="mem_adapter")(mem_emb)
        mem_processed = RMSNorm(self.embed_dim)(mem_processed)

        # Summary Injection (Хураангуйг шингээх)
        alpha     = jax.nn.sigmoid(self.param("alpha_gate", nn.initializers.constant(_logit(self.alpha_init)), (self.embed_dim,)))
        ssum_proj = nn.Dense(self.embed_dim, use_bias=False)(ssum)
        x = x + (ssum_proj[:, None, :] * alpha[None, None, :])

        # KV-Bank байгуулах
        # Дараалал : [Memory (O) | Engram (T) | Current (T)]
        kv_seq = jnp.concatenate([mem_processed, engram_emb, x], axis=1)

        # Mask бэлдэх
        # Causal Mask (Гурвалжин)
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        mem_mask    = jnp.ones((T, O), dtype=bool)
        engram_mask = causal_mask
        full_mask   = jnp.concatenate([mem_mask, engram_mask, causal_mask], axis=1)

        # Padding Mask
        valid_curr = (tokens_w != pad_id)
        valid_mem  = jnp.ones((B, O), dtype=bool)
        valid_eng  = valid_curr
        valid_k    = jnp.concatenate([valid_mem, valid_eng, valid_curr], axis=1)

        mask = full_mask[None, None, :, :] & valid_k[:, None, None, :]

        # RoPE давтамж бэлдэх
        total_kv_len = O + T + T
        freqs_cis    = precompute_freqs_cis(self.embed_dim // self.num_heads, total_kv_len + 32)

        # Transformer Layers
        curr_x = x
        for i in range(self.num_layers):
            curr_x = TransformerBlock(self.embed_dim, self.num_heads, name=f"b{i}")(
                curr_x, mask=mask, kv=kv_seq, freqs_cis=freqs_cis, deterministic=self.deterministic
            )

        curr_x = RMSNorm(self.embed_dim)(curr_x)

        # Дараагийн төлөвийг хадгалах (Update State)
        new_mem_emb = curr_x[:, -O:, :]
        new_mem_ids = tokens_w[:, -O:]

        # Summary шинэчлэх логик
        out_mask = (tokens_w != pad_id).astype(jnp.float32)[:, :, None]
        win_sum  = jnp.sum(curr_x * out_mask, axis=1) / (jnp.sum(out_mask, axis=1) + 1e-6)

        lam      = jax.nn.sigmoid(self.param("lambda_gate", nn.initializers.constant(_logit(self.lambda_init)), (self.embed_dim,)))
        new_ssum = (ssum * lam[None, :]) + (win_sum * (1.0 - lam[None, :]))

        logits = nn.Dense(self.vocab_size)(curr_x)
        return (new_mem_emb, new_mem_ids, new_ssum), logits


# ҮНДСЭН МОДЕЛЬ

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
    engram_num_heads  : int

    @nn.compact
    def __call__(self, tokens_long, deterministic=True):
        # tokens_long: (B, Total_Seq_Len)
        B, N    = tokens_long.shape
        W, O, S = self.window_len, self.overlap, self.window_len - self.overlap

        # Текстийг жижиг цонхнуудад (Windows) хуваах
        n_win = 1
        if N > W:
            n_win = int(math.ceil((N - W) / S)) + 1

        total_len_needed = W + (n_win - 1) * S
        pad_amount       = max(0, total_len_needed - N)

        tokens_pad = jnp.pad(tokens_long, ((0, 0), (0, pad_amount)), constant_values=pad_id)
        starts     = (jnp.arange(n_win) * S).astype(jnp.int32)
        
        # vmap ашиглан цонхнуудыг зэрэгцүүлэн үүсгэх
        windows    = jax.vmap(lambda s: jax.lax.dynamic_slice(tokens_pad, (0, s), (B, W)))(starts)

        # JAX Scan ашиглан рекуррент гүйдлийг үүсгэх
        ScanCell = nn.scan(
            CWRRTEWindowCell,
            variable_broadcast = "params",
            split_rngs         = {"params": False, "dropout": True},
            in_axes            = 0,
            out_axes           = 0
        )

        # Анхны төлөвүүдийг (Carry) 0-ээр дүүргэх
        init_mem_emb = jnp.zeros((B, O, self.embed_dim))
        init_mem_ids = jnp.zeros((B, O), dtype=jnp.int32)
        init_ssum    = jnp.zeros((B, self.embed_dim))

        # Scan ажиллуулах (цонхнуудаар давтах)
        _, logits_windows = ScanCell(
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
            engram_num_heads  = self.engram_num_heads,
            deterministic     = deterministic
        )((init_mem_emb, init_mem_ids, init_ssum), windows)

        # Цонхнуудын гаралтыг буцааж нэг урт дараалал болгох
        out = logits_windows[0]
        if n_win > 1:
            rest = logits_windows[1:, :, O:, :].transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            out  = jnp.concatenate([out, rest], axis=1)

        return out[:, :N, :]


# МАШИН СУРГАЛТ БОЛОН GENERATE ХИЙХ ФУНКЦУУД

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
    engram_ngram_n    = engram_ngram_n,
    engram_num_heads  = engram_num_heads
)

@jax.jit
def train_step(state, batch, rng):
    """Нэг сургалтын алхам (Loss тооцох + Gradient шинэчлэх)"""
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(p):
        logits = model.apply({"params": p}, batch[:, :-1], deterministic=False, rngs={"dropout": dropout_rng})
        labels = batch[:, 1:]
        logits = logits[:, :labels.shape[1], :]

        loss_t = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask   = (labels != pad_id).astype(jnp.float32)
        loss   = jnp.sum(loss_t * mask) / (jnp.sum(mask) + 1e-6)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_rng

@jax.jit
def predict_step_jit(params, fixed_input):
    """Текст үүсгэхэд ашиглах хурдасгасан функц"""
    return model.apply({"params": params}, fixed_input, deterministic=True)

def generate(params, prompt, gen_len=100, temp=0.8):
    """Өгөгдсөн эхлэлээс үргэлжлүүлэн текст зохиох"""
    token_ids = list(encode_text(prompt))
    if token_ids[-1] == eos_id: token_ids.pop()

    max_len = sft_long_seq_len
    print(f"Текст үүсгэж байна: '{prompt}'")

    for _ in range(gen_len):
        curr_len = len(token_ids)
        if curr_len >= max_len: break

        pad_len = max_len - curr_len
        inp_np  = np.array(token_ids + [pad_id] * pad_len, dtype=np.int32)
        inp_jax = jnp.array([inp_np])

        logits = predict_step_jit(params, inp_jax)
        next_token_logits = logits[0, curr_len - 1, :]

        # Тусгай токенуудыг сонгохгүй байхаар тохируулах
        next_token_logits = next_token_logits.at[pad_id].set(-1e9)
        next_token_logits = next_token_logits.at[bos_id].set(-1e9)

        probs     = np.exp(np.array(next_token_logits) / temp)
        probs_sum = np.sum(probs)
        if probs_sum == 0 or np.isnan(probs_sum): probs = np.ones_like(probs) / len(probs)
        else: probs /= probs_sum

        next_id = np.random.choice(len(probs), p=probs)
        token_ids.append(next_id)

        if next_id == eos_id: break

    return decode_ids(token_ids)


# ENGRAM DEBUG HELPERS (Gate + Engram RMS шалгах)

def _find_engram_subtree(params):
    """Engram-ийн параметрийн модыг олно (Debugging)"""
    def _rec(node, path):
        if hasattr(node, "items"):
            keys = list(node.keys())
            if ("engram_table" in keys) and ("engram_gate" in keys):
                return node, path
            for k, v in node.items():
                sub, sub_path = _rec(v, f"{path}/{k}" if path else str(k))
                if sub is not None:
                    return sub, sub_path
        return None, None
    return _rec(params, "")

def _sigmoid_gate_stats(engram_subtree):
    """Gate-ийн дундаж утгыг шалгах"""
    gate = jax.nn.sigmoid(engram_subtree["engram_gate"])
    return float(jnp.mean(gate)), float(jnp.min(gate)), float(jnp.max(gate))

def _engram_rms_from_batch(engram_subtree, tokens_w, mem_ids):
    """Engram гаралтын RMS-ийг хэмжих (Scaling зөв эсэхийг шалгахад хэрэгтэй)"""
    engram_mod = MultiHeadEngram(
        vocab_size   = vocab_size,
        embed_dim    = embed_dim,
        memory_size  = engram_vocab_size,
        ngram_n      = engram_ngram_n,
        num_heads    = engram_num_heads,
        dropout_rate = engram_dropout
    )
    out = engram_mod.apply({"params": engram_subtree}, tokens_w, mem_ids, deterministic=True)
    return float(jnp.sqrt(jnp.mean(out * out)))


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps"      , type=int  , default=sft_total_steps )
    parser.add_argument("--seq-len"    , type=int  , default=sft_long_seq_len)
    parser.add_argument("--batch"      , type=int  , default=sft_batch_size  )
    parser.add_argument("--loss-freq"  , type=int  , default=sft_loss_freq   )
    parser.add_argument("--sample-freq", type=int  , default=sft_sample_freq )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CWRRTE : Cross-Window Recurrent Transformer + Multi-Head Engram")
    print(f"  Steps: {args.steps} | Batch: {args.batch} | SeqLen: {args.seq_len}")
    print(f"  Engram Size: {engram_vocab_size} | N-gram: {engram_ngram_n} | Heads: {engram_num_heads}")
    print("="*60 + "\n")

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Моделийн параметрүүдийг үүсгэх
    dummy_in  = jnp.zeros((1, args.seq_len), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_in, deterministic=True)
    params    = variables["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Моделийн нийт параметр: {param_count/1e6:.2f}M")

    # Optimizer тохиргоо
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=1e-6, peak_value=sft_learning_rate,
                warmup_steps=sft_warmup_steps, decay_steps=args.steps, end_value=1e-6
            ),
            weight_decay=weight_decay
        )
    )

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    start_time = time.time()

    # Машин сургалтын процесс
    for step in range(1, args.steps + 1):
        starts    = np.random.randint(0, len(corpus_ids) - args.seq_len - 1, args.batch)
        batch_np  = np.stack([corpus_ids[s : s + args.seq_len + 1] for s in starts])
        batch_jax = jnp.asarray(batch_np, dtype=jnp.int32)

        state, loss, rng = train_step(state, batch_jax, rng)

        if step % args.loss_freq == 0:
            dt = time.time() - start_time

            # Engram gate статистик
            engram_subtree, engram_path = _find_engram_subtree(state.params)
            if engram_subtree is not None:
                g_mean, g_min, g_max = _sigmoid_gate_stats(engram_subtree)
                print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s | EngramGate(mean/min/max): {g_mean:.4f}/{g_min:.4f}/{g_max:.4f} | Path: {engram_path}")
            else:
                print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s | EngramGate: NOT FOUND")

        if step % args.sample_freq == 0:
            # Engram output RMS шалгалт
            engram_subtree, _ = _find_engram_subtree(state.params)
            if engram_subtree is not None:
                tokens_w = batch_jax[:, :cwr_window_len]
                mem_ids  = jnp.zeros((batch_jax.shape[0], cwr_overlap), jnp.int32)
                e_rms    = _engram_rms_from_batch(engram_subtree, tokens_w, mem_ids)
                print(f"EngramOut RMS (first window, deterministic): {e_rms:.6f}")

            print("\n" + "-"*40)
            sample_text = generate(state.params, sample_prompt_text, gen_len=sample_gen_len, temp=sample_temp)
            print(f"ГАРСАН ҮР ДҮН: {sample_text}")
            speak_async(sample_text, voice=tts_voice, speed=tts_speed, amp=tts_amp, enabled=tts_enabled)
            print("-"*40 + "\n")

    print("Сургалт амжилттай дууслаа!")


if __name__ == "__main__":
    main()
