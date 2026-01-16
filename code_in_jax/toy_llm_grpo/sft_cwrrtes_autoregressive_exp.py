#
#  CWRRTES : A Cross-Window Recurrent Transformer with Dual-State Engram Memory + Salience-Gated Recurrent Memory
#
#  АРХИТЕКТУРЫН ТАЙЛБАР:
#   Энэ модель нь Recurrent (Дараалсан) болон Retrieval (Хайлт хийх) аргуудыг
#   хослуулан, урт контексттэй ажиллах чадвартай архитектурын туршилт юм.
#   Дараах үндсэн дөрвөн бүрэлдэхүүн хэсгүүдээс тогтоно:
#
#   1. Recurrent Memory (Short-Term Context):
#      Өмнөх цонхны сүүлийн хэсгийг (overlap) дараагийн цонх руу шууд дамжуулж,
#      дарааллын тасралтгүй байдлыг хангана.
#
#   2. Dual-State Global Summary (Long-Term Context):
#      Текстийн ерөнхий агуулгыг хадгалахдаа мартах буюу leaky асуудлыг шийдэхийн
#      тулд санах ойг хоёр түвшинд хувааж ажиллуулна:
#      - Fast Summary : Ойрын харилцан яриа, богино үйл явдлыг хадгална.
#      - Slow Summary : Surprise-Based Update ашиглан зөвхөн гэнэтийн, шинэ
#        мэдээлэл ирсэн үед л шинэчлэгдэж, агуулгын ерөнхий үйл явдлыг удаан хадгална.
#
#      - Token-Conditional Injection : Хураангуйг бүх токенд хүчээр наахгүй,
#        токен бүр өөрт хэрэгтэй эсэхийг шийдэх Gate-ээр дамжуулж авна.
#
#      - SGRM (Multi-Head Salience-Gated Recurrent Memory):
#        - Санах ойг олон толгой (Multi-Head) болгож, текстийн өөр өөр
#          шинж чанарыг (Fact vs Style) тус тусад нь шүүж хадгална.
#        - Write Strength Gate : Цонх бүр санах ойд бичих эсэхээ өөрөө шийднэ.
#        - Write Budget        : Санах ойг хэт их эсвэл хэт бага шинэчлэхээс сэргийлсэн regularization (Target Rate ~20%) ашиглагдсан.
#
#   3. Multi-Head Engram Memory (Static Knowledge):
#      DeepSeek аргачлалын дагуу текст дэх тогтмол хэллэг, нэр томьёог (N-grams)
#      нейрон сүлжээгээр биш, хурдан хайлтаар (Lookup Table) олж авна.
#      - Хэш мөргөлдөөнийг арилгахын тулд 4-8 өөр төрлийн анхны тоо ашиглан
#        зэрэгцүүлэн хайлт хийнэ (Vectorized Rolling Hash).
#
#   4. KV-Bank Integration:
#      [Recurrent Overlap | Engram Memory | Current Tokens] гэсэн дарааллаар
#      Attention механизмд орж, өнгөрсөн болон одоог зэрэг харах боломжийг олгоно.
#      - RoPE Alignment : Query болон Key-ийн байршлын зөрүүг засч,
#        Current token-ууд зөв цаг хугацаандаа (position offset) байрлана.
#
#
#  Ашиглагдсан шийдлүүд:
#   - RMSNorm  : LayerNorm-оос хөнгөн бөгөөд сургалтыг тогтворжуулна.
#   - SwiGLU   : Llama-3, DeepSeek зэрэг орчин үеийн загваруудын стандарт MLP.
#   - JAX Scan : Цонх хоорондын рекуррент шилжилтийг өндөр хурдаар гүйцэтгэнэ.
#   - RoPE     : Байршлын векторыг дарааллын эхнээс зөв хуваарилна.
#   - remat    : Activation memory (BPTT) багасгах Gradient Checkpointing.
#
#
#  TTS eSpeak суулгах (текст сонсох):
#   - Ubuntu/WSL:  sudo apt install espeak-ng
#

import os
# Санах ойн хуваарилалтыг оновчтой болгох (Fragmentation багасгах)
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

# CWRRTES Архитектур
cwr_window_len        = 128    # Нэг удаад боловсруулах цонхны урт
cwr_overlap           = 32     # Дараагийн цонх руу дамжуулах overlap урт

# Engram Санах ой (Multi-Head)
engram_vocab_size     = 100000 # Engram хүснэгтийн нийт мөр (Slot count)
engram_ngram_n        = 4      # Хэдэн тэмдэгт харж хэш хийх вэ (N-gram)
engram_num_heads      = 4      # Санах ойг хэдэн толгойгоор зэрэг хайх вэ
engram_dropout        = 0.05   # Санах ойгоос хэт хамаарахаас сэргийлэх dropout

# SGRM (Salience-Gated Recurrent Memory)
sgrm_dropout          = 0.00
sgrm_num_heads        = 4      # SGRM-д зориулсан толгойн тоо (Transformer head-ээс тусдаа)
sgrm_target_rate      = 0.20   # Gate нээлттэй байх дундаж хувь (Regularization target)
sgrm_budget_weight    = 0.1    # Loss дээрх Regularization-ийн жин

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
max_seq_len           = 8192   # (одоогоор ашиглаж буй дээд хязгаар)

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
    x        : (B, T, H, D)
    freq_cis : (1, T, 1, D/2)
    """
    B, T, H, D = x.shape
    x_complex = jax.lax.complex(x[..., 0::2], x[..., 1::2])
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

def slice_freqs_cis(freqs_cis_global, pos_idx):
    """
    Absolute position-оор RoPE slice хийх (Global Cache-ээс).
    freqs_cis_global: (1, MaxLen, 1, Dh/2)
    pos_idx         : (T,) int32 (Absolute index)
    return          : (1, T, 1, Dh/2)
    """
    pos_idx = pos_idx.astype(jnp.int32)
    return jnp.take(freqs_cis_global, pos_idx, axis=1)


# RMSNorm, SwiGLU (Үндсэн блокууд)

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    LayerNorm-ийг бодвол mean (дундаж) тооцохгүй тул хурдан, мөн Deep NN-д gradient урсгалыг тогтвортой байлгана.
    """
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        rms   = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * scale

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    GELU-г орлоно. Gate болон Value салаатай тул логик сэтгэлгээг сайжруулдаг гэлцдэг.
    Llama-3, DeepSeek зэрэг орчин үеийн загваруудын стандарт.
    """
    embed_dim     : int
    expand_factor : int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden_dim = int(self.embed_dim * self.expand_factor)

        gate_val = nn.Dense(hidden_dim * 2, use_bias=False)(x)
        gate, val = jnp.split(gate_val, 2, axis=-1)

        act = nn.silu(gate) * val
        out = nn.Dense(self.embed_dim, use_bias=False)(act)

        return nn.Dropout(0.1, deterministic=deterministic)(out)


# SGRM, SALIENCE-GATED RECURRENT MEMORY

class SalienceWriteHead(nn.Module):
    """
    SGRM:
    - Multi-Head Salience   : Санах ойг олон хүчин зүйл (Head)-д хуваана.
    - NaN-Safe Softmax      : Padding буюу хоосон цонхны хамгаалалт.
    - Learnable Temperature : Шүүлтүүрийн мэдрэг байдлыг сурна.
    - Per-Head Gating       : Толгой бүр бичих эсэхээ тусдаа шийднэ.
    """
    embed_dim    : int
    num_heads    : int = 4
    dropout_rate : float = 0.0

    @nn.compact
    def __call__(self, x, mask_bool, deterministic=True):
        B, T, D  = x.shape
        head_dim = D // self.num_heads

        x_heads = x.reshape(B, T, self.num_heads, head_dim)

        temp_param  = self.param("temp", nn.initializers.ones, (self.num_heads,))
        temperature = nn.softplus(temp_param) + 0.3

        salience_proj = nn.Dense(self.num_heads, name="salience_logits")(x) # (B, T, H)
        logits        = salience_proj / temperature[None, None, :]

        mask_expanded = mask_bool[:, :, None]
        safe_logits   = jnp.where(mask_expanded, logits, -1e9)

        max_logits = jax.lax.stop_gradient(jnp.max(safe_logits, axis=1, keepdims=True))
        exps       = jnp.exp(safe_logits - max_logits)

        exps     = jnp.where(mask_expanded, exps, 0.0)
        sum_exps = jnp.sum(exps, axis=1, keepdims=True) + 1e-6
        weights  = exps / sum_exps

        write_vec_heads = jnp.sum(weights[:, :, :, None] * x_heads, axis=1) # (B, H, Dh)

        gate_logits = nn.Dense(1, name="gate_proj")(write_vec_heads) # (B, H, 1)

        window_valid  = jnp.any(mask_bool, axis=1)[:, None, None]
        write_u_heads = nn.sigmoid(gate_logits) * window_valid.astype(jnp.float32)

        write_vec = write_vec_heads.reshape(B, D)
        write_vec = RMSNorm(self.embed_dim)(write_vec)

        if self.dropout_rate > 0.0:
            write_vec = nn.Dropout(self.dropout_rate, deterministic=deterministic)(write_vec)

        write_u_expanded = jnp.tile(write_u_heads, (1, 1, head_dim)).reshape(B, D)

        return write_vec, write_u_expanded, write_u_heads.squeeze(-1)


# MULTI-HEAD ENGRAM MEMORY (DeepSeek)

class MultiHeadEngram(nn.Module):
    """
    Олон толгойтой Engram санах ой.
    Сайжруулалт:
      1. Head бүр өөр анхны тоонууд ашиглаж зэрэгцээ хайлт хийнэ.
      2. JAX vmap ашиглан толгойнуудыг vectorization хийнэ.
      3. Үр дүнг нэгтгэхдээ Gating ашиглана.
    """
    vocab_size   : int
    embed_dim    : int
    memory_size  : int   = 100000
    ngram_n      : int   = 4
    num_heads    : int   = 4
    dropout_rate : float = 0.05

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, "Embed dim must be divisible by heads"
        self.head_dim = self.embed_dim // self.num_heads

        self.memory_table = self.param(
            "engram_table",
            nn.initializers.normal(stddev=0.02),
            (self.memory_size, self.num_heads, self.head_dim)
        )

        self.gate_logit = self.param(
            "engram_gate",
            nn.initializers.constant(-2.0),
            (self.num_heads, self.head_dim)
        )

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
        B, W = current_ids.shape
        O    = prev_ids_overlap.shape[1]
        assert O >= (self.ngram_n - 1)

        current_ids      = jnp.where(current_ids      == pad_id, 0, current_ids)
        prev_ids_overlap = jnp.where(prev_ids_overlap == pad_id, 0, prev_ids_overlap)

        full_seq  = jnp.concatenate([prev_ids_overlap, current_ids], axis=1).astype(jnp.uint32)
        start_idx = O

        hash_sums = jnp.zeros((B, W, self.num_heads), dtype=jnp.uint32)

        for i in range(self.ngram_n):
            s_start  = start_idx - i
            s_end    = full_seq.shape[1] - i
            chunk    = full_seq[:, s_start:s_end]                              # (B, W)
            p_vec    = self.primes[:, i]                                       # (H,)
            hash_sums = hash_sums + (chunk[:, :, None] * p_vec[None, None, :])

        lookup_indices = (hash_sums % self.memory_size).astype(jnp.int32)      # (B, W, H)

        table_h = jnp.transpose(self.memory_table, (1, 0, 2))
        idx_h   = jnp.transpose(lookup_indices, (2, 0, 1))

        def _gather(tbl, idx):
            return tbl[idx]                                                    # (B, W, Dh)

        got_h = jax.vmap(_gather, in_axes=(0, 0), out_axes=0)(table_h, idx_h)  # (H, B, W, Dh)
        retrieved = jnp.transpose(got_h, (1, 2, 0, 3))                         # (B, W, H, Dh)

        gate = jax.nn.sigmoid(self.gate_logit)                                 # (H, Dh)
        out  = retrieved * gate[None, None, :, :]

        out  = out.reshape(B, W, self.embed_dim)
        out  = nn.Dropout(self.dropout_rate, deterministic=deterministic)(out)
        return out


# TRANSFORMER BLOCKS (Pre-RMSNorm & SwiGLU)

class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        if kv is None:
            kv = x

        B, Tq, _ = x.shape
        _, Tk, _ = kv.shape
        head_dim = self.embed_dim // self.num_heads

        q = nn.Dense(self.embed_dim, name="q_proj")(x)
        k = nn.Dense(self.embed_dim, name="k_proj")(kv)
        v = nn.Dense(self.embed_dim, name="v_proj")(kv)

        q = q.reshape(B, Tq, self.num_heads, head_dim)
        k = k.reshape(B, Tk, self.num_heads, head_dim)
        v = v.reshape(B, Tk, self.num_heads, head_dim)

        if freqs_cis is not None:
            if isinstance(freqs_cis, tuple):
                f_q, f_k = freqs_cis
                q = apply_rope(q, f_q)
                k = apply_rope(k, f_k)
            else:
                q = apply_rope(q, freqs_cis[:, :Tq])
                k = apply_rope(k, freqs_cis[:, :Tk])

        q = q.transpose(0, 2, 1, 3) # (B, H, Tq, D)
        k = k.transpose(0, 2, 1, 3) # (B, H, Tk, D)
        v = v.transpose(0, 2, 1, 3) # (B, H, Tk, D)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)

        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(0.1, deterministic=deterministic)(attn_probs)

        out = jnp.matmul(attn_probs, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)

        return nn.Dense(self.embed_dim, name="out_proj")(out)

class TransformerBlock(nn.Module):
    """
    Pre-RMSNorm бүтэцтэй, MLP нь SwiGLU-тэй.
    Scan / remat ашиглахад зориулагдсан.
    """
    embed_dim     : int
    num_heads     : int
    deterministic : bool = True

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None):
        norm_x  = RMSNorm(self.embed_dim)(x)
        norm_kv = kv if kv is not None else norm_x

        attn_out = CausalSelfAttention(self.embed_dim, self.num_heads)(
            norm_x,
            mask          = mask,
            kv            = norm_kv,
            freqs_cis     = freqs_cis,
            deterministic = self.deterministic
        )
        x = x + attn_out

        norm_x2 = RMSNorm(self.embed_dim)(x)
        mlp_out = SwiGLU(self.embed_dim)(
            norm_x2,
            deterministic=self.deterministic
        )
        x = x + mlp_out

        return x


# CWRRTES RECURRENT CELL (RoPE, SGRM)

class CWRRTESWindowCell(nn.Module):
    vocab_size        : int
    embed_dim         : int
    num_layers        : int
    num_heads         : int
    window_len        : int
    overlap           : int
    engram_vocab_size : int
    engram_ngram_n    : int
    engram_num_heads  : int
    deterministic     : bool = True

    @nn.compact
    def __call__(self, carry, tokens_w, win_start, freqs_cis_global):

        mem_emb, mem_ids, ssum_fast, ssum_slow = carry

        B, T = tokens_w.shape
        O    = self.overlap

        # Оролтын Embedding
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)

        # Injection: Контекстийг оруулахын өмнө тогтворжуулна (Stable Injection)
        global_ctx     = jnp.concatenate([ssum_fast, ssum_slow], axis=-1)         # (B, 2*D)
        ctx_proj       = nn.Dense(self.embed_dim, name="ctx_read_proj")(global_ctx)
        ctx_proj       = RMSNorm(self.embed_dim)(ctx_proj)

        # Token-Conditional Gate
        ctx_broadcast  = jnp.broadcast_to(ctx_proj[:, None, :], x.shape)
        concat_input   = jnp.concatenate([x, ctx_broadcast], axis=-1)
        injection_gate = nn.sigmoid(nn.Dense(self.embed_dim, name="inject_gate")(concat_input))

        x = x + (ctx_proj[:, None, :] * injection_gate)

        # Engram Retrieval & Processing
        engram_emb = MultiHeadEngram(
            vocab_size   = self.vocab_size,
            embed_dim    = self.embed_dim,
            memory_size  = self.engram_vocab_size,
            ngram_n      = self.engram_ngram_n,
            num_heads    = self.engram_num_heads,
            dropout_rate = engram_dropout
        )(tokens_w, mem_ids, deterministic=self.deterministic)

        engram_emb    = RMSNorm(self.embed_dim)(engram_emb)
        mem_processed = nn.Dense(self.embed_dim, name="mem_adapter")(mem_emb)
        mem_processed = RMSNorm(self.embed_dim)(mem_processed)

        # ROPE ALIGNMENT (ABSOLUTE POSITIONS)
        #  Энд Global Cache-ээс тухайн цонхонд харгалзах давтамжийг тасдаж авна.
        #  base_pos = O (mem-ийг negative index-ээс хамгаалах шилжлэг)
        #  - Curr pos: base_pos + win_start + [0..T-1]
        #  - Mem  pos: base_pos + win_start - O + [0..O-1]
        # ----------------------------------------------------------------------

        base_pos  = O
        win_start = win_start.astype(jnp.int32)

        pos_curr = base_pos + win_start + jnp.arange(T, dtype=jnp.int32)                 # (T,)
        pos_mem  = base_pos + win_start - O + jnp.arange(O, dtype=jnp.int32)             # (O,)

        f_mem  = slice_freqs_cis(freqs_cis_global, pos_mem)                              # (1, O, 1, Dh/2)
        f_curr = slice_freqs_cis(freqs_cis_global, pos_curr)                             # (1, T, 1, Dh/2)

        # KV: [mem(O) | engram(T) | curr(T)]
        f_kv     = jnp.concatenate([f_mem, f_curr, f_curr], axis=1)                      # (1, O+2T, 1, Dh/2)
        f_curr_q = f_curr

        # Mask бэлдэх
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        full_mask   = jnp.concatenate([jnp.ones((T, O), dtype=bool), causal_mask, causal_mask], axis=1)

        valid_curr  = (tokens_w != pad_id)
        valid_mem   = jnp.ones((B, O), dtype=bool)
        valid_k     = jnp.concatenate([valid_mem, valid_curr, valid_curr], axis=1)
        mask        = full_mask[None, None, :, :] & valid_k[:, None, None, :]

        # Transformer давхаргууд
        curr_x = x
        for i in range(self.num_layers):
            kv_seq = jnp.concatenate([mem_processed, engram_emb, curr_x], axis=1)

            blk = nn.remat(TransformerBlock)(
                self.embed_dim,
                self.num_heads,
                deterministic = self.deterministic,
                name          = f"b{i}"
            )

            curr_x = blk(
                curr_x,
                mask      = mask,
                kv        = kv_seq,
                freqs_cis = (f_curr_q, f_kv)
            )

        curr_x = RMSNorm(self.embed_dim)(curr_x)

        # Update Logic
        new_mem_emb = curr_x[:, -O:, :]
        new_mem_ids = tokens_w[:, -O:]
        valid_mask  = (tokens_w != pad_id)

        write_vec, write_u_expanded, write_u_raw = SalienceWriteHead(
            self.embed_dim, num_heads=sgrm_num_heads, dropout_rate=sgrm_dropout
        )(curr_x, valid_mask, deterministic=self.deterministic)

        # FAST Summary Update (SGRM + EMA)
        gate_fast     = nn.sigmoid(self.param("gate_fast", nn.initializers.constant(0.0), (self.embed_dim,)))
        new_ssum_fast = (ssum_fast * gate_fast) + (write_vec * (1.0 - gate_fast) * write_u_expanded)

        # SLOW Summary Update (Novelty Gated + SGRM)
        def cosine_sim(a, b):
            a_n = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
            b_n = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-6)
            return jnp.sum(a_n * b_n, axis=-1, keepdims=True)

        sim        = cosine_sim(jax.lax.stop_gradient(write_vec), jax.lax.stop_gradient(ssum_slow))
        novelty    = 1.0 - sim
        lambda_eff = jnp.clip(0.99 - (0.5 * novelty), 0.5, 0.999)

        new_ssum_slow = (ssum_slow * lambda_eff) + (write_vec * (1.0 - lambda_eff) * write_u_expanded)

        logits = nn.Dense(self.vocab_size)(curr_x)

        return (new_mem_emb, new_mem_ids, new_ssum_fast, new_ssum_slow), (logits, write_u_raw)


# ҮНДСЭН МОДЕЛЬ

class CWRRTESTransformer(nn.Module):
    vocab_size        : int
    embed_dim         : int
    num_layers        : int
    num_heads         : int
    window_len        : int
    overlap           : int
    engram_vocab_size : int
    engram_ngram_n    : int
    engram_num_heads  : int

    @nn.compact
    def __call__(self, tokens_long, deterministic=True):
        # tokens_long: (B, Total_Seq_Len)
        B, N    = tokens_long.shape
        W, O, S = self.window_len, self.overlap, self.window_len - self.overlap

        n_win = 1
        if N > W:
            n_win = int(math.ceil((N - W) / S)) + 1

        total_len_needed = W + (n_win - 1) * S
        pad_amount       = max(0, total_len_needed - N)

        tokens_pad = jnp.pad(tokens_long, ((0, 0), (0, pad_amount)), constant_values=pad_id)
        starts     = (jnp.arange(n_win) * S).astype(jnp.int32)

        # windows: (n_win, B, W)
        windows = jax.vmap(lambda s: jax.lax.dynamic_slice(tokens_pad, (0, s), (B, W)))(starts)

        # ROPE GLOBAL CACHE (ONE-TIME PRECOMPUTE)

        head_dim = self.embed_dim // self.num_heads
        base_pos = O
        extra    = 64
        rope_len = int(base_pos + total_len_needed + extra)

        freqs_cis_global = precompute_freqs_cis(head_dim, rope_len)

        # JAX SCAN WRAPPER (CLOSURE)

        # Бид Wrapper дотор cell-ээ үүсгэх тул config-уудаа хадгалж авна.
        cfg_vocab_size        = self.vocab_size
        cfg_embed_dim         = self.embed_dim
        cfg_num_layers        = self.num_layers
        cfg_num_heads         = self.num_heads
        cfg_window_len        = self.window_len
        cfg_overlap           = self.overlap
        cfg_engram_vocab_size = self.engram_vocab_size
        cfg_engram_ngram_n    = self.engram_ngram_n
        cfg_engram_num_heads  = self.engram_num_heads
        cfg_deterministic     = deterministic

        class ScanLoopWrapper(nn.Module):
            @nn.compact
            def __call__(self, carry, inputs):
                # inputs tuple-ийг задална
                tokens_w, win_start = inputs

                # Жинхэнэ cell-ээ дуудна
                cell = CWRRTESWindowCell(
                    vocab_size        = cfg_vocab_size,
                    embed_dim         = cfg_embed_dim,
                    num_layers        = cfg_num_layers,
                    num_heads         = cfg_num_heads,
                    window_len        = cfg_window_len,
                    overlap           = cfg_overlap,
                    engram_vocab_size = cfg_engram_vocab_size,
                    engram_ngram_n    = cfg_engram_ngram_n,
                    engram_num_heads  = cfg_engram_num_heads,
                    deterministic     = cfg_deterministic
                )

                # freqs_cis_global нь гаднах scope-оос харагдана (Closure)
                return cell(carry, tokens_w, win_start, freqs_cis_global)

        # JAX Scan ашиглах
        # in_axes=0 гэдэг нь (windows, starts) tuple-ийн бүх элементийг 0-р тэнхлэгээр scan хийнэ гэсэн үг.
        ScanCell = nn.scan(
            ScanLoopWrapper,
            variable_broadcast = "params",
            split_rngs         = {"params": False, "dropout": True},
            in_axes            = 0,
            out_axes           = 0
        )

        init_mem_emb   = jnp.zeros((B, O, self.embed_dim))
        init_mem_ids   = jnp.zeros((B, O), dtype=jnp.int32)
        init_ssum_fast = jnp.zeros((B, self.embed_dim))
        init_ssum_slow = jnp.zeros((B, self.embed_dim))

        # freqs_cis_global-ийг аргументэд бичих шаардлагагүй.
        _, (logits_windows, aux_windows) = ScanCell(name="scan_main")(
            (init_mem_emb, init_mem_ids, init_ssum_fast, init_ssum_slow),
            (windows, starts)
        )

        out = logits_windows[0]
        if n_win > 1:
            rest = logits_windows[1:, :, O:, :].transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            out  = jnp.concatenate([out, rest], axis=1)

        return out[:, :N, :], aux_windows


# МАШИН СУРГАЛТ БОЛОН GENERATE ХИЙХ ФУНКЦУУД

model = CWRRTESTransformer(
    vocab_size        = vocab_size,
    embed_dim         = embed_dim,
    num_layers        = num_layers,
    num_heads         = num_heads,
    window_len        = cwr_window_len,
    overlap           = cwr_overlap,
    engram_vocab_size = engram_vocab_size,
    engram_ngram_n    = engram_ngram_n,
    engram_num_heads  = engram_num_heads
)

@jax.jit
def train_step(state, batch, rng):
    """Нэг сургалтын алхам (Loss + Write Budget Regularization)"""
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(p):
        logits, write_gates = model.apply(
            {"params": p}, batch[:, :-1], deterministic=False, rngs={"dropout": dropout_rng}
        )
        labels = batch[:, 1:]
        logits = logits[:, :labels.shape[1], :]

        loss_t = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask   = (labels != pad_id).astype(jnp.float32)
        text_loss = jnp.sum(loss_t * mask) / (jnp.sum(mask) + 1e-6)

        mean_gate   = jnp.mean(write_gates)
        budget_loss = (mean_gate - sgrm_target_rate) ** 2

        total_loss  = text_loss + (sgrm_budget_weight * budget_loss)
        return total_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_rng

@jax.jit
def predict_step_jit(params, fixed_input):
    """Текст үүсгэхэд ашиглах хурдасгасан функц"""
    return model.apply({"params": params}, fixed_input, deterministic=True)[0]

def generate(params, prompt, gen_len=100, temp=0.8, ctx_len=768):
    """
    Өгөгдсөн эхлэлээс үргэлжлүүлэн текст зохиох
    Sliding Window ашиглана (хэт урт контекстийг тасдах)
    """
    token_ids = list(encode_text(prompt))
    if token_ids[-1] == eos_id:
        token_ids.pop()

    print(f"Текст үүсгэж байна: '{prompt}'")

    for _ in range(gen_len):
        curr_len = len(token_ids)
        # Хэрэв дээд хязгаарт тулбал зогсох
        if curr_len >= sft_long_seq_len:
            break

        # Sliding context logic
        start = max(0, curr_len - ctx_len)
        ctx   = token_ids[start:curr_len]

        inp_np  = np.array(ctx, dtype=np.int32)
        inp_jax = jnp.array([inp_np])

        logits = predict_step_jit(params, inp_jax)                  # (1, len(ctx), V)
        next_token_logits = logits[0, len(ctx) - 1, :]

        next_token_logits = next_token_logits.at[pad_id].set(-1e9)
        next_token_logits = next_token_logits.at[bos_id].set(-1e9)

        probs     = np.exp(np.array(next_token_logits) / temp)
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


# ENGRAM DEBUG HELPERS

def _find_engram_subtree(params):
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
    gate = jax.nn.sigmoid(engram_subtree["engram_gate"])
    return float(jnp.mean(gate)), float(jnp.min(gate)), float(jnp.max(gate))


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
    print("  CWRRTES : Cross-Window Recurrent Transformer + Multi-Head Engram + SGRM")
    print("  Feature : Global RoPE Cache + Absolute Pos Slice + Scan Wrapper")
    print(f"  Steps: {args.steps} | Batch: {args.batch} | SeqLen: {args.seq_len}")
    print(f"  Engram Size: {engram_vocab_size} | SGRM Heads: {sgrm_num_heads}")
    print("="*60 + "\n")

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    dummy_in  = jnp.zeros((1, args.seq_len), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_in, deterministic=True)
    params    = variables["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Моделийн нийт параметр: {param_count/1e6:.2f}M")

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value   = 1e-6,
                peak_value   = sft_learning_rate,
                warmup_steps = sft_warmup_steps,
                decay_steps  = args.steps,
                end_value    = 1e-6
            ),
            weight_decay=weight_decay
        )
    )

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    start_time = time.time()

    for step in range(1, args.steps + 1):
        starts    = np.random.randint(0, len(corpus_ids) - args.seq_len - 1, args.batch)
        batch_np  = np.stack([corpus_ids[s : s + args.seq_len + 1] for s in starts])
        batch_jax = jnp.asarray(batch_np, dtype=jnp.int32)

        state, loss, rng = train_step(state, batch_jax, rng)

        if step % args.loss_freq == 0:
            dt = time.time() - start_time

            engram_subtree, engram_path = _find_engram_subtree(state.params)
            gate_info = ""
            if engram_subtree is not None:
                g_mean, _, _ = _sigmoid_gate_stats(engram_subtree)
                gate_info = f"| EG_Mean: {g_mean:.3f}"

            print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s {gate_info}")

        if step % args.sample_freq == 0:
            print("\n" + "-"*40)
            sample_text = generate(state.params, sample_prompt_text, gen_len=sample_gen_len, temp=sample_temp)
            print(f"ГАРСАН ҮР ДҮН: {sample_text}")
            speak_async(sample_text, voice=tts_voice, speed=tts_speed, amp=tts_amp, enabled=tts_enabled)
            print("-"*40 + "\n")

    print("Сургалт амжилттай дууслаа!")


if __name__ == "__main__":
    main()