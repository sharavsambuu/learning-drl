#
#  CWRRTESOUP : Cross-Window Recurrent Transformer with Episodic Slot Memory & Engram Injection
#
#  АРХИТЕКТУРЫН ДЭЛГЭРЭНГҮЙ ТАЙЛБАР 
#
#  Энэ модель нь урт текстийг уншиж, өмнөх мэдээллээ мартахгүйгээр боловсруулах
#  чадвартай Hybrid Memory архитектур юм. Стандарт Transformer-ууд контекст цонх дүүрэхэд
#  өмнөх мэдээллээ гээдэг бол, энэ модель нь чухал мэдээллийг санах ойн үүрнүүдэд (Slots)
#  хадгалж авч үлдэх боломжтой.
#
#
#  1. EPISODIC SLOT MEMORY (Episodic буюу Үйл явдлын санах ой)
#     Энэ бол моделийн хувьд hard disk нь юм. Мэдээллийг [K, Dim] хэмжээтэй матрицад хадгална.
#
#   - Бүтэц (Slot Anatomy):
#      - Key      (Хаяг   )  : Мэдээллийг эргүүлж хайхад ашиглах вектор (Index).
#      - Value    (Агуулга)  : Хадгалах гэж буй бодит мэдээллийн вектор.
#      - Strength (Хүч    )  : Энэ мэдээлэл хэр чухал вэ? (0.0 - 1.0). Чухал биш бол хурдан мартана.
#      - Age      (Нас    )  : Энэ үүрийг ашиглалгүй хэр удаж байна вэ? (LRU Logic).
#                              Хуучирсан мэдээллийг шинэ мэдээллээр дарж бичнэ.
#
#   - Competitive Write (Өрсөлдөөнт бичилт & Smearing асуудлыг шийддэг):
#      - Асуудал (Softmax Smearing): 
#        Энгийн Attention механизмаар санах ойд бичихэд
#        бүх үүр рүү бага багаар зэрэг бичсээр байгаад, хэсэг хугацааны дараа
#        бүх үүрний мэдээлэл холилдож бүдгэрч эхэлдэг.
#      - Шийдэл нь (ST Top-k): 
#        Straight-Through Estimator ашиглах.
#           - Forward Pass: 
#             Хамгийн сул эсвэл тохиромжтой ТOP-1 (эсвэл Top-k) үүрийг
#             сонгож, зөвхөн тэр үүр рүү хатуу (Hard) бичилт хийнэ.
#             Ингэснээр мэдээлэл холилдохгүй, тод, цэгцтэй хадгалагдана.
#           - Backward Pass: 
#             Машин сургалтын үед Gradient таслахгүйн тулд үүнийг
#             Softmax-аар бичсэн мэтээр тооцоолж, жингээ шинэчилнэ.
#
#  2. ENGRAM MEMORY (Тогтмол хэллэгийн санах ой)
#
#   - Асуудал: 
#     Аливаа хэлэнд "Once upon a time", "United States of America" гэх мэт
#     тогтмол давтагддаг хэллэгүүд (N-grams) маш их байдаг. Эдгээрийг заавал
#     Attention механизмаар тооцоолох нь өөрөө нөөц үрсэн хэрэг юм.
#   - Шийдэл: 
#     Эдгээр хэллэгүүдийг Hashing (Rolling Hash) аргаар тооцоолж,
#     том хүснэгтээс шууд бэлэн вектор хэлбэрээр татаж авна.
#   - Injection: 
#     Татаж авсан мэдээллээ KV Cache руу биш, шууд Residual Stream руу нийлүүлнэ. 
#     Ингэснээр санах ойн хурдыг нэмэгдүүлэх боломжтой.
#
#  3. RECURRENT STATE & CONTROL (Дамжих төлөв ба Удирдлага)
#     Модель нь цонх (Window) хооронд шилжихдээ дараах мэдээллийг тээж явна:
#
#   - Fast Summary : 
#     Сүүлийн хэдэн өгүүлбэрийн нарийн ширийн мэдээлэл.
#   - Slow Summary : 
#     Текстийн ерөнхий сэдэв, гол агуулга (Topic vector).
#   - Initialization: 
#     Машин сургалтын эхэнд эдгээр векторууд 0 байвал Normalization хийх үед NaN алдаа гардаг.
#     Иймд эхлэхдээ маш бага хэмжээний noise-той (random) үүсгэж эхлүүлнэ.
#   - Recall Gate (Хэзээ санах вэ?): 
#     Модель алхам бүрт санах ойг ухах шаардлагагүй.
#     Одоогийн уншиж буй текст нь ойлгомжгүй эсвэл шинэ сэдэв байвал Gate нээгдэж, 
#     санах ойноос мэдээллээ татна.
#   - Salience Writer (Юуг бичих вэ?): 
#     Текст дэх бүх үгийг хадгалах боломжгүй.
#     SGRM механизм нь хамгийн чухал буюу Salient мэдээллийг шүүж аваад хадгална.
#
#  4. ТЕХНИК ШИЙДЛҮҮД
#   - JAX Scan : 
#     For-loop ашиглахын оронд JAX-ийн компиляцид тохирсон Scan ашиглаж, машин сургалтын хурдыг эрс нэмэгдүүлдэг.
#   - Global RoPE : 
#     Байршлын мэдээллийг (Positional Encoding) цонх дамжсан хэдий ч абсолют байдлаар зөв тооцоолно.
#   - Contrastive Loss: 
#     Модель санах ойд бичсэн зүйлээ буцааж олж чадаж байгаа эсэхийг шалгаж, санах ойн найдвартай байдлыг хангана.
#
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

# CWRRTE Архитектур (Цонхны тохиргоо)
cwr_window_len        = 128    # Нэг удаад боловсруулах цонхны урт
cwr_overlap           = 32     # Дараагийн цонх руу дамжуулах overlap урт

# Engram Санах ой (N-gram Hashing Lookup)
engram_vocab_size     = 100000 # Хэш хүснэгтийн нийт мөр (Slot count)
engram_ngram_n        = 4      # Хэдэн тэмдэгт харж хэш хийх вэ (N-gram)
engram_num_heads      = 4      # Санах ойг хэдэн толгойгоор зэрэг хайх вэ
engram_dropout        = 0.05   # Overfit-ээс сэргийлэх dropout

# SGRM (Salience-Gated Recurrent Memory - Бичих удирдлага)
sgrm_dropout          = 0.00
sgrm_num_heads        = 4      # Бичих үйлдлийг шийдэх толгойн тоо
sgrm_target_rate      = 0.20   # Gate нээлттэй байх дундаж хувь (Regularization)
sgrm_budget_weight    = 0.1    # Loss дээрх Regularization-ийн жин

# Episodic Slot Memory (Үндсэн Санах Ой)
epi_num_slots         = 64     # Санах ойн нийт үүрний тоо (Slots)
epi_strength_decay    = 0.995  # Санах ойн хүч цаг хугацааны эрхээр сулрах хурд
epi_age_penalty       = 0.02   # Хуучин мэдээллийг унших магадлалыг бууруулах торгууль
epi_strength_boost    = 0.50   # Чухал мэдээллийг унших магадлалыг нэмэгдүүлэх
epi_write_alpha       = 0.50   # Шинэ мэдээлэл бичих хурд (Soft Update Rate)
epi_min_strength      = 1e-3   # Санах ойн хамгийн бага хүч

# Competitive Write (Scalability)
#   - Soft write нь олон цонх дээр "smearing" үүсгэж, слотын ялгарал алдагддаг.
#   - ST Top-k ашиглавал forward дээр өрсөлдөөнтэй (sparse) бичээд, backward дээр gradient хадгална.
epi_write_temp        = 50.0   # Softmax sharpening
epi_write_topk        = 1      # Top-k слот руу бичих (1 = argmax шиг, гэхдээ gradient дамжуулна)

# Recall Gate (Санах ойг шагайх эсэх)
recall_target_rate    = 0.15   # Санах ойг унших дундаж давтамж
recall_budget_weight  = 0.05   # Үүнийг зохицуулах Loss-ийн жин

# Contrastive Recall (Санах ойн бүрэн бүтэн байдал)
contrastive_weight    = 0.05   # Credit Assignment Loss-ийн жин

# Сургалтын тохиргоо (SFT)
sft_total_steps       = 5000
sft_long_seq_len      = 1024   # Нэг багцад орох текстийн урт
sft_batch_size        = 8
sft_learning_rate     = 5e-4
sft_warmup_steps      = 100

# Лог болон явцын хяналт
sft_loss_freq         = 10
sft_sample_freq       = 100
sample_gen_len        = 256
sample_temp           = 0.8
sample_prompt_text    = "Once upon a time there was a little robot. "

# eSpeak TTS
tts_enabled           = True
tts_voice             = "en"
tts_speed             = 165
tts_amp               = 120
tts_max_chars         = 400

# Моделийн хэмжээ
num_layers            = 6
num_heads             = 8
embed_dim             = 512
max_seq_len           = 8192

# Optimizer
max_grad_norm         = 1.0
weight_decay          = 0.01

# Random seed тохиргоо
np.random.seed(seed)
random.seed(seed)


# ESPEAK NON-BLOCKING TTS 

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
    if not t: return
    try:
        p = subprocess.Popen(
            [_ESPEAK_BIN, "-v", voice, "-s", str(speed), "-a", str(amp), "--stdin"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        p.stdin.write((t + "\n").encode("utf-8", errors="ignore"))
        p.stdin.close()
    except Exception: pass


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


# ТУСЛАХ ФУНКЦУУД (RoPE, Math)

def apply_rope(x, freq_cis):
    """
    Rotary Positional Embeddings (RoPE).
    Векторыг комплекс тоон талбарт эргүүлэх замаар байршлын мэдээллийг шингээнэ.
    """
    B, T, H, D = x.shape
    x_complex = jax.lax.complex(x[..., 0::2], x[..., 1::2])
    x_rotated = x_complex * freq_cis
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1).reshape(B, T, H, D)
    return x_out

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    """RoPE-ийн давтамжийн матрицыг урьдчилан тооцоолох"""
    freqs     = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t         = jnp.arange(max_len)
    freqs     = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis[None, :, None, :]

def slice_freqs_cis(freqs_cis_global, pos_idx):
    """
    Global Cache-ээс тухайн цонхны байршилд (Absolute Position)
    харгалзах давтамжийг тасдаж авах.
    """
    pos_idx = pos_idx.astype(jnp.int32)
    return jnp.take(freqs_cis_global, pos_idx, axis=1)

def cosine_sim(a, b):
    """Вектор хоорондын ижил төстэй байдлыг хэмжих (Cosine Similarity)"""
    a_n = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
    b_n = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-6)
    return jnp.sum(a_n * b_n, axis=-1)


# BASIC LAYERS (RMSNorm, SwiGLU)

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Тооцооллын хувьд LayerNorm-оос хөнгөн бөгөөд тогтвортой.
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
    Swish-Gated Linear Unit (MLP Layer).
    Gate mechanism ашиглан мэдээллийг шүүж дамжуулдаг орчин үеийн стандарт.
    """
    embed_dim     : int
    expand_factor : int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden_dim = int(self.embed_dim * self.expand_factor)
        gate_val   = nn.Dense(hidden_dim * 2, use_bias=False)(x)
        gate, val  = jnp.split(gate_val, 2, axis=-1)
        act        = nn.silu(gate) * val
        out        = nn.Dense(self.embed_dim, use_bias=False)(act)
        return nn.Dropout(0.1, deterministic=deterministic)(out)


# EPISODIC SLOT MEMORY (CORE LOGIC)

class MemoryRecallGate(nn.Module):
    """
    When-to-look Policy:
    Загвар одоогийн цонхны мэдээлэл дээр үндэслэн санах ойг ухах шаардлагатай юу?
    гэдгийг шийддэг Gate (Хаалга).
    """
    embed_dim : int

    @nn.compact
    def __call__(self, q_win, writer_strength, novelty, deterministic=True):
        # q_win : Цонхны ерөнхий агуулга
        x = jnp.concatenate([q_win, writer_strength, novelty], axis=-1)
        h = nn.Dense(self.embed_dim)(x)
        h = nn.silu(h)
        return nn.sigmoid(nn.Dense(1)(h))

class EpisodicSlotReader(nn.Module):
    """
    Targeted Read:
    Санах ойн олон слотуудаас (Slot Bank) хамгийн хэрэгтэйг нь шүүж уншина.
    Томьёо: 
        Logits = Similarity + Strength_Boost - Age_Penalty
    """
    embed_dim : int

    @nn.compact
    def __call__(self, q_win, epi_keys, epi_vals, epi_age, epi_strength, deterministic=True):
        # Cosine Similarity тооцох (Query vs Memory Keys)
        qn  = q_win / (jnp.linalg.norm(q_win, axis=-1, keepdims=True) + 1e-6)
        kn  = epi_keys / (jnp.linalg.norm(epi_keys, axis=-1, keepdims=True) + 1e-6)
        sim = jnp.sum(kn * qn[:, None, :], axis=-1)

        # Strength болон Age-ийг тооцоололд оруулах
        logits = sim + (epi_strength_boost * jnp.log(jnp.clip(epi_strength, 1e-3, 1e9))) - (epi_age_penalty * epi_age)

        # Хоосон слотуудыг уншихгүй байх (Masking)
        logits = logits + ((epi_strength > 1e-3).astype(jnp.float32) - 1.0) * 1e3

        # Softmax Attention-аар жигнэсэн дундаж авах
        w    = jax.nn.softmax(logits, axis=-1)
        read = jnp.sum(w[:, :, None] * epi_vals, axis=1)
        read = RMSNorm(self.embed_dim)(read)

        return read, w, logits

class EpisodicSlotWriter(nn.Module):
    """
    SOFT WRITER + COMPETITIVE WRITE (ST Top-k):
    - Softmax нь gradient хадгална, гэхдээ удаан хугацаанд slot smearing үүсгэнэ.
    - ST Top-k ашиглавал forward дээр sparse (өрсөлдөөнтэй) бичээд, backward дээр soft gradient дамжуулна.
    """
    embed_dim : int
    num_slots : int

    @nn.compact
    def __call__(self, write_key, write_val, write_strength, epi_keys, epi_vals, epi_age, epi_strength, deterministic=True):
        B, K, D = epi_keys.shape

        # Similarity
        wk_n = write_key / (jnp.linalg.norm(write_key, axis=-1, keepdims=True) + 1e-6)
        ek_n = epi_keys  / (jnp.linalg.norm(epi_keys , axis=-1, keepdims=True) + 1e-6)
        sim  = jnp.sum(ek_n * wk_n[:, None, :], axis=-1)                                         # (B, K)

        # Soft Addressing
        write_logits = sim * epi_write_temp
        write_w_soft = jax.nn.softmax(write_logits, axis=-1)                                     # (B, K)

        # Competitive Write: Top-k hard mask (Forward)
        k = int(epi_write_topk)
        if k <= 1:
            top_idx  = jnp.argmax(write_w_soft, axis=-1)  # (B,)
            write_w_hard = jax.nn.one_hot(top_idx, K, dtype=jnp.float32)                         # (B, K)
        else:
            topk_idx     = jnp.argsort(write_w_soft, axis=-1)[:, -k:]                            # (B, k)
            write_w_hard = jnp.sum(jax.nn.one_hot(topk_idx, K, dtype=jnp.float32), axis=1)       # (B, K)
            write_w_hard = write_w_hard / (jnp.sum(write_w_hard, axis=-1, keepdims=True) + 1e-6)

        # Straight-Through, forward=hard, backward=soft
        write_w = jax.lax.stop_gradient(write_w_hard - write_w_soft) + write_w_soft              # (B, K)

        # Update Rates
        ws   = jnp.clip(write_strength, 0.0, 1.0)                                                # (B, 1)
        ws_k = ws                                                                                # (B, 1) -> broadcast to (B, K)

        eff_rate = write_w * ws_k * epi_write_alpha                                              # (B, K)
        rate_exp = eff_rate[:, :, None]                                                          # (B, K, 1)

        # Content Update (Interpolation)
        keys_new = (1.0 - rate_exp) * epi_keys + rate_exp * write_key[:, None, :]
        vals_new = (1.0 - rate_exp) * epi_vals + rate_exp * write_val[:, None, :]
        keys_new = keys_new / (jnp.linalg.norm(keys_new, axis=-1, keepdims=True) + 1e-6)

        # Age Update (Stable)
        age_new = epi_age + 1.0
        age_new = age_new * (1.0 - write_w)

        # Strength Update (Bounded)
        str_new = epi_strength * epi_strength_decay
        str_new = str_new + (write_w * ws_k) * (1.0 - str_new)
        str_new = jnp.clip(str_new, epi_min_strength, 1.0)

        # Debug info
        best_sim = jnp.max(sim, axis=-1)
        slot_idx = jnp.argmax(write_w_hard, axis=-1)

        return keys_new, vals_new, age_new, str_new, slot_idx, best_sim



# MODULES (SGRM, ENGRAM, TRANSFORMER)

class WindowSalienceWriter(nn.Module):
    """
    SGRM Writer:
    Цонхны бүх токенуудаас хамгийн чухал хэсгүүдийг ялгаж авна.
    """
    embed_dim    : int
    num_heads    : int   = 4
    dropout_rate : float = 0.0

    @nn.compact
    def __call__(self, x, mask_bool, deterministic=True):
        B, T, D = x.shape
        x_heads = x.reshape(B, T, self.num_heads, D // self.num_heads)

        # Learnable Temperature
        temp    = nn.softplus(self.param("temp", nn.initializers.ones, (self.num_heads,))) + 0.3
        logits  = nn.Dense(self.num_heads)(x) / temp[None, None, :]

        mask    = mask_bool[:, :, None]
        safe_l  = jnp.where(mask, logits, -1e9)
        exps    = jnp.exp(safe_l - jax.lax.stop_gradient(jnp.max(safe_l, axis=1, keepdims=True)))
        exps    = jnp.where(mask, exps, 0.0)
        weights = exps / (jnp.sum(exps, axis=1, keepdims=True) + 1e-6)

        vec     = RMSNorm(self.embed_dim)(jnp.sum(weights[:, :, :, None] * x_heads, axis=1).reshape(B, D))
        gate    = nn.sigmoid(nn.Dense(1)(vec.reshape(B, self.num_heads, -1))) * jnp.any(mask_bool, axis=1)[:, None, None]

        write_strength = jnp.mean(gate.squeeze(-1), axis=-1, keepdims=True)
        return vec, write_strength, gate.squeeze(-1)

class NgramEngramMemory(nn.Module):
    """
    Engram Memory:
    Тогтмол хэллэгүүдийг (N-grams) нейрон сүлжээгээр биш, Hashing аргаар шууд санах ойн хүснэгтээс татаж авна.
    """
    vocab_size: int; embed_dim: int; memory_size: int=100000; ngram_n: int=4; num_heads: int=4

    def setup(self):
        self.head_dim = self.embed_dim // self.num_heads
        self.table    = self.param("engram_table", nn.initializers.normal(0.02)  , (self.memory_size, self.num_heads, self.head_dim))
        self.gate     = self.param("engram_gate" , nn.initializers.constant(-2.0), (self.num_heads, self.head_dim))
        ps = []; base = 131
        for h in range(self.num_heads):
            x, r = base + h * 1009, []
            for _ in range(self.ngram_n): r.append(x); x = (x * 31) + 1
            ps.append(r)
        self.primes = jnp.array(ps, dtype=jnp.uint32)

    @nn.compact
    def __call__(self, curr, prev, deterministic=True):
        B, W  = curr.shape; O = prev.shape[1]
        seq   = jnp.concatenate([jnp.where(prev==pad_id, 0, prev), jnp.where(curr==pad_id, 0, curr)], 1).astype(jnp.uint32)
        h_sum = jnp.zeros((B, W, self.num_heads), dtype=jnp.uint32)

        # Rolling Hash
        for i in range(self.ngram_n):
            h_sum += (seq[:, O-i:seq.shape[1]-i, None] * self.primes[None, None, i, :])

        idx = (h_sum % self.memory_size).astype(jnp.int32)
        got = jax.vmap(
            lambda t, i: t[i],
            in_axes=(0, 0),
            out_axes=0
        )(self.table.transpose(1, 0, 2), idx.transpose(2, 0, 1)).transpose(1, 2, 0, 3)

        # Gate (Per-head, per-dim)
        gate = jax.nn.sigmoid(self.gate)[None, None, :, :]  # (1,1,H,hd)
        out  = got * gate
        out  = out.reshape(B, W, self.embed_dim)
        out  = nn.Dropout(engram_dropout, deterministic=deterministic)(out)
        return out

class TransformerBlock(nn.Module):
    """
    Standard Transformer Block:
    Attention болон MLP давхаргуудыг агуулна. Remat ашиглан VRAM хэмнэнэ.
    """
    embed_dim     : int
    num_heads     : int
    deterministic : bool=True

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None):
        def attn(x, kv):
            B, T, _ = x.shape; H = self.num_heads; D = self.embed_dim // H
            q, k, v = [nn.Dense(self.embed_dim)(t).reshape(B, -1, H, D) for t in (x, kv, kv)]
            if freqs_cis is not None:
                if isinstance(freqs_cis, tuple):
                    q, k = apply_rope(q, freqs_cis[0]), apply_rope(k, freqs_cis[1])
                else:
                    q, k = apply_rope(q, freqs_cis[:, :T]), apply_rope(k, freqs_cis[:, :kv.shape[1]])

            w = jnp.matmul(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 3, 1)) / math.sqrt(D)
            if mask is not None: w = jnp.where(mask, w, -1e9)

            p = nn.Dropout(0.1, deterministic=self.deterministic)(jax.nn.softmax(w))
            return jnp.matmul(p, v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)

        norm_x  = RMSNorm(self.embed_dim)(x)
        norm_kv = kv if kv is not None else RMSNorm(self.embed_dim)(x)
        x = x + nn.Dense(self.embed_dim)(attn(norm_x, norm_kv))
        x = x + SwiGLU(self.embed_dim)(RMSNorm(self.embed_dim)(x), deterministic=self.deterministic)
        return x


# CWRRTE WINDOW CELL

class CWRRTEWindowCell(nn.Module):
    vocab_size        : int
    embed_dim         : int
    num_layers        : int
    num_heads         : int
    window_len        : int
    overlap           : int
    engram_vocab_size : int
    engram_ngram_n    : int
    engram_num_heads  : int
    max_seq_len       : int
    deterministic     : bool=True

    @nn.compact
    def __call__(self, carry, tokens_w, win_start, freqs_cis_global):
        # State (Carry)-г задлах
        mem_emb, mem_ids, ssum_fast, ssum_slow, epi_keys, epi_vals, epi_age, epi_strength, last_write_key, pos_base = carry
        B, T = tokens_w.shape; O = self.overlap

        # Input Embedding
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)

        # Legacy Injection (Global Summary оруулах)
        g_ctx = RMSNorm(self.embed_dim)(nn.Dense(self.embed_dim)(jnp.concatenate([ssum_fast, ssum_slow], -1)))
        gate  = nn.sigmoid(nn.Dense(self.embed_dim)(jnp.concatenate([x, jnp.broadcast_to(g_ctx[:, None], x.shape)], -1)))
        x     = x + g_ctx[:, None] * gate

        # Memory Read (Engram + Legacy + Episodic)
        engram = RMSNorm(self.embed_dim)(
            NgramEngramMemory(
                self.vocab_size,
                self.embed_dim,
                self.engram_vocab_size,
                self.engram_ngram_n,
                self.engram_num_heads
            )(tokens_w, mem_ids, deterministic=self.deterministic)
        )
        mem_pr = RMSNorm(self.embed_dim)(nn.Dense(self.embed_dim)(mem_emb))

        # Engram-г KV рүү хийхгүй (Scalability)
        #   - KV уртыг багасгаж, нэг цонхны тооцоог хөнгөлнө.
        #   - Engram-г шууд x урсгалд inject хийнэ.
        x = x + nn.Dense(self.embed_dim, name="engram_inj")(engram)

        # Episodic Read (Targeted)
        q_win   = jnp.mean(RMSNorm(self.embed_dim)(x), 1)
        novelty = 1.0 - cosine_sim(jax.lax.stop_gradient(q_win), jax.lax.stop_gradient(ssum_slow))[:, None]
        g_rec   = MemoryRecallGate(self.embed_dim)(q_win, jnp.ones((B, 1))*0.5, novelty)
        epi_r, _, epi_logits = EpisodicSlotReader(self.embed_dim)(q_win, epi_keys, epi_vals, epi_age, epi_strength)

        # Retrieved info-г нэмэх (Gated)
        x = x + nn.Dense(self.embed_dim)(epi_r * jax.lax.stop_gradient(g_rec))[:, None]

        # RoPE Alignment (Global Slice)
        f_mem = slice_freqs_cis(freqs_cis_global, pos_base - O + jnp.arange(O))
        f_cur = slice_freqs_cis(freqs_cis_global, pos_base + jnp.arange(T))

        # KV = [mem_pr (O), curr (T)] тул f_kv = [f_mem, f_cur]
        f_kv  = jnp.concatenate([f_mem, f_cur], 1)

        # Attention Mask (KV урт = O + T)
        mask = (
            jnp.concatenate(
                [jnp.ones((T, O)), jnp.tril(jnp.ones((T, T)))],
                1
            ).astype(bool)[None, None]
            &
            jnp.concatenate(
                [jnp.ones((B, O)), tokens_w!=pad_id],
                1
            ).astype(bool)[:, None, None]
        )

        # Transformer Layers
        curr = x
        for i in range(self.num_layers):
            kv   = jnp.concatenate([mem_pr, curr], 1)
            curr = nn.remat(TransformerBlock)(
                self.embed_dim,
                self.num_heads,
                deterministic=self.deterministic,
                name=f"b{i}"
            )(curr, mask, kv, (f_cur, f_kv))
        curr = RMSNorm(self.embed_dim)(curr)

        # Writers (Санах ойг шинэчлэх)
        w_vec, w_str, w_heads = WindowSalienceWriter(self.embed_dim, sgrm_num_heads)(
            curr, tokens_w!=pad_id, deterministic=self.deterministic
        )

        # Episodic Write (Soft + Competitive)
        w_key = nn.Dense(self.embed_dim)(w_vec)
        w_val = nn.Dense(self.embed_dim)(w_vec)
        w_key = w_key / (jnp.linalg.norm(w_key, axis=-1, keepdims=True) + 1e-6)

        ek2, ev2, ea2, es2, s_idx, b_sim = EpisodicSlotWriter(self.embed_dim, epi_num_slots)(
            w_key, w_val, w_str, epi_keys, epi_vals, epi_age, epi_strength
        )

        # Legacy Updates
        g_fast = nn.sigmoid(self.param("gf", nn.initializers.zeros, (self.embed_dim,)))
        sf2    = ssum_fast * g_fast + w_vec * (1-g_fast) * w_str
        lam    = jnp.clip(
            0.99 - 0.5 * (1.0 - cosine_sim(jax.lax.stop_gradient(w_vec), jax.lax.stop_gradient(ssum_slow))[:, None]),
            0.5, 0.999
        )
        ss2    = ssum_slow * lam + w_vec * (1-lam) * w_str

        # Contrastive Loss Target (Би юу бичнэсээ санаж байна уу?)
        prev_target = jnp.argmax(
            jnp.sum(
                epi_keys * (last_write_key / (jnp.linalg.norm(last_write_key, -1, keepdims=True)+1e-6))[:, None],
                -1
            ),
            -1
        )
        nll = -jax.nn.log_softmax(epi_logits, -1)[jnp.arange(B), prev_target] * (jnp.linalg.norm(last_write_key, -1) > 0)

        # Carry болон Output бэлдэх
        new_carry = (
            curr[:, -O:], tokens_w[:, -O:],
            sf2, ss2,
            ek2, ev2, ea2, es2,
            w_key, pos_base + (self.window_len - O)
        )
        aux_out = (nn.Dense(self.vocab_size)(curr), (w_heads, g_rec, nll, s_idx, b_sim))

        return new_carry, aux_out


# MAIN TRANSFORMER (JAX SCAN WRAPPER)

class CWRRTETransformer(nn.Module):
    vocab_size        : int
    embed_dim         : int
    num_layers        : int
    num_heads         : int
    window_len        : int
    overlap           : int
    engram_vocab_size : int
    engram_ngram_n    : int
    engram_num_heads  : int
    max_seq_len       : int

    @nn.compact
    def __call__(self, tokens, deterministic=True):
        B, N  = tokens.shape; W, O = self.window_len, self.overlap; S = W - O
        n_win = math.ceil((N - W) / S) + 1 if N > W else 1

        # Window Slicing
        pad_t  = jnp.pad(tokens, ((0, 0), (0, max(0, W + (n_win - 1) * S - N))), constant_values=pad_id)
        starts = (jnp.arange(n_win) * S).astype(jnp.int32)
        wins   = jax.vmap(lambda s: jax.lax.dynamic_slice(pad_t, (0, s), (B, W)))(starts)

        # Global RoPE Cache
        freqs = precompute_freqs_cis(self.embed_dim // self.num_heads, int(self.max_seq_len + 512))

        # Scan хийхэд зориулсан closure (Static args)
        cfg = {k: getattr(self, k) for k in ["vocab_size", "embed_dim", "num_layers", "num_heads", "window_len", "overlap", "engram_vocab_size", "engram_ngram_n", "engram_num_heads", "max_seq_len"]}
        cfg["deterministic"] = deterministic

        class ScanLoopWrapper(nn.Module):
            @nn.compact
            def __call__(self, carry, inputs):
                tokens_w, win_start = inputs
                cell = CWRRTEWindowCell(**cfg)
                return cell(carry, tokens_w, win_start, freqs)

        # Scan Initialization
        ScanCell = nn.scan(
            ScanLoopWrapper,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=0, out_axes=0
        )

        
        ik = jax.random.PRNGKey(99) # Local init key
        k1, k2, k3, k4, k5 = jax.random.split(ik, 5)

        init = (
            jnp.zeros((B, O, self.embed_dim)),                                 # Mem Emb
            jnp.zeros((B, O), dtype=jnp.int32),                                # Mem IDs
            
            # Summaries ийг noise тэйгээр эхлүүлэх 
            jax.random.normal(k1, (B, self.embed_dim)) * 0.01,                 # Fast Summary
            jax.random.normal(k2, (B, self.embed_dim)) * 0.01,                 # Slow Summary
            
            # Episodic Keys/Vals-г noise тэйгээр эхлүүлэх
            jax.random.normal(k3, (B, epi_num_slots, self.embed_dim)) * 0.02,  # Epi Keys
            jax.random.normal(k4, (B, epi_num_slots, self.embed_dim)) * 0.01,  # Epi Vals
            
            jnp.ones((B, epi_num_slots))*1e3,                                  # Epi Age
            jnp.zeros((B, epi_num_slots)),                                     # Epi Strength
            
            # Last Write Key ийг noise тэйгээр эхлүүлэх
            jax.random.normal(k5, (B, self.embed_dim)) * 0.01,                 # Last Write Key
            
            jnp.array(0, dtype=jnp.int32)                                      # Position Base
        )

        _, (log, aux) = ScanCell(name="scan_main")(init, (wins, starts))

        # Гаралтыг нэгтгэх
        out = log[0]
        if n_win > 1:
            out = jnp.concatenate([out, log[1:, :, O:].transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)], 1)
        return out[:, :N], aux


# TRAINING & GENERATION LOOP

model = CWRRTETransformer(vocab_size, embed_dim, num_layers, num_heads, cwr_window_len, cwr_overlap, engram_vocab_size, engram_ngram_n, engram_num_heads, max_seq_len)

@jax.jit
def train_step(state, batch, rng):
    d_rng, n_rng = jax.random.split(rng)
    def loss_fn(p):
        l, aux = model.apply({"params": p}, batch[:, :-1], deterministic=False, rngs={"dropout": d_rng})
        ce     = optax.softmax_cross_entropy_with_integer_labels(l[:, :batch.shape[1]-1], batch[:, 1:])
        mask   = (batch[:, 1:] != pad_id).astype(jnp.float32)

        # Loss бүрэлдэхүүн хэсгүүд
        text_loss     = jnp.sum(ce * mask) / (jnp.sum(mask) + 1e-6)
        write_budget  = (jnp.mean(aux[0]) - sgrm_target_rate)**2
        recall_budget = (jnp.mean(aux[1]) - recall_target_rate)**2
        contrastive   = jnp.mean(aux[2])

        return text_loss + \
                (sgrm_budget_weight   * write_budget ) + \
                (recall_budget_weight * recall_budget) + \
                (contrastive_weight   * contrastive  )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss, n_rng

@jax.jit
def predict_step(params, inputs):
    return model.apply({"params": params}, inputs, deterministic=True)[0]

def generate(params, prompt, gen_len=100, temp=0.8):
    """
    GENERATE with PADDING:
    JAX Re-compilation-аас сэргийлж тогтмол хэмжээтэй (Padded) оролт ашиглана.
    """
    ids = encode_text(prompt)
    if ids[-1] == eos_id: ids.pop()
    print(f"Текст үүсгэж байна: '{prompt}'")

    # Pre-allocate fixed buffer
    MAX_INFER = sft_long_seq_len

    for _ in range(gen_len):
        curr = len(ids)
        if curr >= MAX_INFER: break

        # Padded Input бэлдэх
        inp = np.full((MAX_INFER,), pad_id, dtype=np.int32)

        # Sliding Window (Хэт урт үед)
        real_ctx = ids[-MAX_INFER:] if curr > MAX_INFER else ids
        L = len(real_ctx)
        inp[:L] = real_ctx

        # JIT Call
        logits = predict_step(params, jnp.array([inp]))

        # Sampling
        next_l = logits[0, L-1, :].at[pad_id].set(-1e9).at[bos_id].set(-1e9)
        probs  = np.exp(np.array(next_l) / temp)
        probs_sum = np.sum(probs)
        if probs_sum == 0 or np.isnan(probs_sum): probs = np.ones_like(probs)/len(probs)
        else: probs /= probs_sum

        nxt = np.random.choice(len(probs), p=probs)
        ids.append(nxt)
        if nxt == eos_id: break

    return decode_ids(ids)

# DEBUG HELPER
def _find_engram_subtree(params):
    def _rec(node, path):
        if hasattr(node, "items"):
            if "engram_table" in node.keys(): return node, path
            for k, v in node.items():
                sub, sp = _rec(v, f"{path}/{k}" if path else str(k))
                if sub: return sub, sp
        return None, None
    return _rec(params, "")

def _sigmoid_gate_stats(subtree):
    g = jax.nn.sigmoid(subtree["engram_gate"])
    return float(jnp.mean(g)), float(jnp.min(g)), float(jnp.max(g))


# MAIN EXECUTION

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=sft_total_steps)
    args = p.parse_args()

    print(f"\n{'='*60}\n  CWRRTES : Cross-Window Recurrent Transformer\n  Architecture : Episodic Slots + Soft-Write (ST Top-k) + Engram Inject\n{'='*60}\n")

    rng = jax.random.PRNGKey(seed); r1, r2 = jax.random.split(rng)

    dummy_input = jnp.zeros((1, sft_long_seq_len), dtype=jnp.int32)
    params      = model.init(r1, dummy_input)["params"]

    print(f"Моделийн нийт параметр: {sum(x.size for x in jax.tree_util.tree_leaves(params))/1e6:.2f}M")

    opt = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(1e-6, sft_learning_rate, sft_warmup_steps, args.steps, 1e-6),
            weight_decay=weight_decay
        )
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    t0 = time.time()
    for s in range(1, args.steps+1):
        idx   = np.random.randint(0, len(corpus_ids)-sft_long_seq_len-1, sft_batch_size)
        batch = jnp.array(np.stack([corpus_ids[i:i+sft_long_seq_len+1] for i in idx]))

        state, loss, r2 = train_step(state, batch, r2)

        if s % sft_loss_freq == 0:
            est, _ = _find_engram_subtree(state.params)
            gate_info = f"| EG_Mean: {_sigmoid_gate_stats(est)[0]:.3f}" if est else ""
            print(f"Step {s:5d} | Loss: {loss:.4f} | Time: {time.time()-t0:.1f}s {gate_info}")

        if s % sft_sample_freq == 0:
            print("\n" + "-"*40)
            sample_text = generate(state.params, sample_prompt_text, sample_gen_len, sample_temp)
            print(f"Үүсгэсэн текст: {sample_text}")
            speak_async(sample_text, voice=tts_voice, speed=tts_speed, amp=tts_amp, enabled=tts_enabled)
            print("\n" + "-"*40)

    print("Сургалт амжилттай дууслаа!")

if __name__ == "__main__":
    main()
