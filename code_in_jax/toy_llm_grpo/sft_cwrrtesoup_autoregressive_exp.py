#
#
#  CWRRTES + Episodic Slots : Cross-Window Recurrent Transformer with Hybrid Memory Architecture
#
#  АРХИТЕКТУРЫН ДЭЛГЭРЭНГҮЙ ТАЙЛБАР:
#   Энэ код нь Long-Context буюу урт хэмжээний текстийг боловсруулахдаа зөвхөн 
#   хураангуйлах (Summary) биш, өнгөрсөн үйл явдлыг нарийвчлан хадгалах
#   Slot-Based Episodic Memory-ийн туршилт юм.
#
#   ҮНДСЭН БҮРЭЛДЭХҮҮН ХЭСГҮҮД:
#
#   1. Recurrent Memory (Short-Term & Overlap):
#      - Цонх (Window) хооронд шилжихдээ сүүлийн хэсгийг (Overlap) болон
#        хураангуй векторыг (State) дамжуулж, тасралтгүй байдлыг хангана.
#
#   2. Addressable Episodic Slot Memory (The Soup Ingredient):
#      - Уламжлалт нэг вектор (Summary) санах ойн оронд [K, Dim] хэмжээтэй
#        олон тусдаа үүрнүүдээс (Slot) бүрдэх банк ашиглана.
#      - Slot бүр нь: 
#           Key      (Хайх түлхүүр              ), 
#           Value    (Агуулга                   ), 
#           Strength (Чухал чанар               ),
#           Age      (Хэр удаан ашиглагдаагүй вэ) 
#        гэсэн мэдээллүүдийг хадгална.
#
#   3. Memory Operations (Унших/Бичих Логик):
#      - Targeted Read: 
#        Одоогийн цонхны агуулга (Query) дээр үндэслэн санах ойн слотуудаас 
#        Similarity + Strength - Age гэсэн томьёогоор шүүж уншина.
#      - Recall Gate (When-to-look): 
#        Санах ойг байнга ухахгүйн тулд "Одоо надад санах ой хэрэгтэй юу?" 
#        гэдгийг шийдэх Gate (Хаалт).
#      - Controlled Write: 
#        Шинэ мэдээллийг бичихдээ:
#           - Хэрэв төстэй слот байвал -> Merge (Нэгтгэх)
#           - Төстэй зүйл байхгүй бол  -> Overwrite LRU (Хамгийн хуучирсаныг дарах)
#
#   4. Multi-Head Engram Memory (Static Knowledge):
#      - Текстэд байнга таардаг хэллэгүүдийг (N-grams) нейрон сүлжээгээр биш,
#        Hashing аргаар шууд санах ойгоос татаж авна (DeepSeek аргачлал).
#
#   5. Objectives & Regularization:
#      - Text Loss          : Дараагийн үгийг таах.
#      - SGRM Budget        : Санах ойд хэт их эсвэл хэт бага бичихээс сэргийлэх.
#      - Recall Budget      : Санах ойг шаардлагагүй үед нээхээс сэргийлэх.
#      - Contrastive Recall : Би дөнгөж сая бичсэн зүйлээ эргээд олж чадах уу?
#                             гэсэн асуултад хариулах замаар санах ойн бүрэн 
#                             бүтэн байдлыг хангах Loss.
#
#
#  Ашиглагдсан Технологиуд:
#   - JAX/Flax : Өндөр хурдны тооцоолол, XLA compilation.
#   - RoPE     : Rotary Positional Embeddings (Байршлын мэдээлэл).
#   - SwiGLU   : Сайжруулсан MLP бүтэц.
#   - remat    : Gradient Checkpointing (VRAM хэмнэнэ).
#
#  
#  Энэ код бол зүгээр туршилт учраас ssum_slow (хуучин арга) болон Episodic Slots (шинэ арга)
#  зэрэгцэн ажиллаж байгаа. Episodic нь Primary буюу үндсэн санах ой болсон гэсэн үг.
#
#
#  TTS eSpeak суулгах 
#   Ubuntu/WS : sudo apt install espeak-ng
#
#


import os
# JAX-ийн санах ойн хуваарилалтыг хязгаарлах (GPU memory fragmentation-оос сэргийлэх)
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
engram_vocab_size     = 100000 # Хэш хүснэгтийн хэмжээ (Slots)
engram_ngram_n        = 4      # Хэдэн үг/тэмдэгтийг нэг багц гэж харах вэ
engram_num_heads      = 4      # Зэрэгцээ хайлтын толгойн тоо
engram_dropout        = 0.05   # Overfit-ээс сэргийлэх

# SGRM (Salience-Gated Recurrent Memory - Бичих механизм)
sgrm_dropout          = 0.00
sgrm_num_heads        = 4      # Бичих үйлдлийг шийдэх толгойн тоо
sgrm_target_rate      = 0.20   # Write хаалганы нээлттэй байх дундаж хувь
sgrm_budget_weight    = 0.1    # Бичих үйлдлийг зохицуулах Loss-ийн жин

# Episodic Slot Memory (Хаяглах боломжтой санах ой)
epi_num_slots         = 64     # Санах ойн үүрний тоо (K)
epi_merge_threshold   = 0.85   # Ижил төстэй гэж үзэх босго (Similarity > 0.85 -> Merge)
epi_strength_decay    = 0.999  # Санах ойн хүч цаг хугацааны эрхээр сулрах хурд
epi_age_penalty       = 0.02   # Хуучин мэдээллийг унших магадлалыг бууруулах торгууль
epi_strength_boost    = 0.50   # Чухал мэдээллийг унших магадлалыг нэмэгдүүлэх
epi_write_alpha       = 0.25   # Key   (Түлхүүр) шинэчлэх хурд (EMA)
epi_write_beta        = 0.25   # Value (Утга   ) шинэчлэх хурд (EMA)
epi_min_strength      = 1e-3   # Санах ойн хамгийн бага хүч

# Recall Gate (Санах ойг шагайх эсэх)
recall_target_rate    = 0.15   # Санах ойг унших дундаж давтамж
recall_budget_weight  = 0.05   # Үүнийг зохицуулах Loss-ийн жин

# Contrastive Recall (Санах ойн уялдаа холбоо)
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

# Моделийн дотоод хэмжээсүүд
num_layers            = 6
num_heads             = 8
embed_dim             = 512
max_seq_len           = 8192

# Optimizer Settings
max_grad_norm         = 1.0
weight_decay          = 0.01

# Random Seed 
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


# ӨГӨГДӨЛ БЭЛТГЭХ (DATASET LOADING & TOKENIZATION)

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

# Тэмдэгт түвшний (Character-level) энгийн tokenizer
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

flat_tokens = []
print(">>> Токенчилж байна...")
for s in all_stories:
    flat_tokens.extend(encode_text(s))
corpus_ids = np.array(flat_tokens, dtype=np.int32)

print(f"Dataset Stats: Vocab={vocab_size}, Total Tokens={len(corpus_ids)}")


# ТУСЛАХ ФУНКЦУУД (RoPE, Math)

def _logit(p):
    """Магадлалыг (0-1) logit (-inf to +inf) руу хөрвүүлэх (Init хийхэд хэрэгтэй)"""
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

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

def cosine_sim(a, b, eps=1e-6):
    """Вектор хоорондын ижил төстэй байдлыг хэмжих (Cosine Similarity)"""
    a_n = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b_n = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return jnp.sum(a_n * b_n, axis=-1)


#  BASIC BLOCKS (RMSNorm, SwiGLU)

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


# ADDRESSABLE MEMORY STACK (EPISODIC SLOTS)
# Моделийн тархины урт хугацааны санах ойн

class MemoryRecallGate(nn.Module):
    """
    When-to-look Policy:
    Загвар одоогийн цонхны мэдээлэл дээр үндэслэн санах ойг ухах шаардлагатай юу?
    гэдгийг шийддэг Gate.
    """
    embed_dim : int

    @nn.compact
    def __call__(self, q_win, writer_strength, novelty, deterministic=True):
        # q_win          : Одоогийн цонхны ерөнхий агуулга
        # writer_strength: Одоо бичигдэж буй мэдээлэл хэр чухал вэ
        # novelty        : Одоогийн мэдээлэл хэр шинэлэг вэ
        x = jnp.concatenate([q_win, writer_strength, novelty], axis=-1)
        h = nn.Dense(self.embed_dim)(x)
        h = nn.silu(h)
        g = nn.sigmoid(nn.Dense(1)(h))  # 0=Хаалттай, 1=Нээлттэй
        return g

class EpisodicSlotReader(nn.Module):
    """
    Targeted Read:
    Санах ойн олон слотуудаас (Slot Bank) хамгийн хэрэгтэйг нь шүүж уншина.
    Томьёо: Logits = Similarity + Strength_Boost - Age_Penalty
    """
    embed_dim : int

    @nn.compact
    def __call__(self, q_win, epi_keys, epi_vals, epi_age, epi_strength, deterministic=True):
        # epi_keys : Санах ойн түлхүүрүүд (Index)
        # epi_vals : Санах ойн жинхэнэ агуулга (Data)
        # epi_age  : Слот бүрийн нас (Хэр удаан ашиглагдаагүй)

        # Cosine Similarity тооцох (Query vs Memory Keys)
        qn  = q_win / (jnp.linalg.norm(q_win, axis=-1, keepdims=True) + 1e-6)
        kn  = epi_keys / (jnp.linalg.norm(epi_keys, axis=-1, keepdims=True) + 1e-6)
        sim = jnp.sum(kn * qn[:, None, :], axis=-1)

        # Strength болон Age-ийг тооцоололд оруулах
        str_log = jnp.log(jnp.clip(epi_strength, epi_min_strength, 1e9))
        logits  = sim + (epi_strength_boost * str_log) - (epi_age_penalty * epi_age)

        # Хоосон/Сул слотуудыг уншихгүй байх (Masking)
        alive   = (epi_strength > epi_min_strength).astype(jnp.float32)
        logits  = logits + (alive - 1.0) * 1e3

        # Softmax Attention-аар жигнэсэн дундаж авах
        w    = jax.nn.softmax(logits, axis=-1)  # (B, K)
        read = jnp.sum(w[:, :, None] * epi_vals, axis=1)  # (B, D)
        read = RMSNorm(self.embed_dim)(read)
        return read, w, logits

class EpisodicSlotWriter(nn.Module):
    """
    Controlled Write:
    Санах ойд мэдээлэл бичих маш нарийн логик:
    - Хэрэв ижил мэдээлэл (Similarity > Threshold) байвал -> Нэгтгэх (Merge Update).
    - Байхгүй бол                                         -> Хамгийн удаан ашиглагдаагүй (LRU) слотыг дарж шинээр бичих.
    """
    embed_dim : int
    num_slots : int

    @nn.compact
    def __call__(self, write_key, write_val, write_strength, epi_keys, epi_vals, epi_age, epi_strength, deterministic=True):
        B, K, D = epi_keys.shape

        # Одоо байгаа түлхүүрүүдтэй ижил төстэй эсэхийг шалгах
        wk_n = write_key / (jnp.linalg.norm(write_key, axis=-1, keepdims=True) + 1e-6)
        ek_n = epi_keys  / (jnp.linalg.norm(epi_keys , axis=-1, keepdims=True) + 1e-6)
        sim  = jnp.sum(ek_n * wk_n[:, None, :], axis=-1)  # (B, K)

        best_idx = jnp.argmax(sim, axis=-1)               # (B,)
        best_sim = jnp.max(sim, axis=-1, keepdims=True)   # (B, 1)

        # Merge эсвэл Overwrite шийдвэр гаргах
        do_merge = (best_sim > epi_merge_threshold).astype(jnp.int32).squeeze(-1)

        # LRU (Least Recently Used) слотыг олох (Хөгшин + Сул слот)
        age_score = epi_age + (1.0 - jnp.clip(epi_strength, 0.0, 1.0)) * 0.01
        lru_idx   = jnp.argmax(age_score, axis=-1)

        # Эцсийн бичих индекс
        slot_idx = jnp.where(do_merge == 1, best_idx, lru_idx)  # (B,)

        # Age Update, Бүх слотын нас нэмэгдэнэ, харин бичсэнийх 0 болно.
        epi_age_new = epi_age + 1.0
        epi_age_new = epi_age_new.at[jnp.arange(B), slot_idx].set(0.0)

        # Strength Update, Decay болон шинэ мэдээллийн хүчийг нэмэх
        epi_strength_new = epi_strength * epi_strength_decay
        ws = jnp.clip(write_strength.squeeze(-1), 0.0, 1.0)
        prev_str = epi_strength_new[jnp.arange(B), slot_idx]
        upd_str  = jnp.clip(prev_str + ws * (1.0 - prev_str), epi_min_strength, 1.0)
        epi_strength_new = epi_strength_new.at[jnp.arange(B), slot_idx].set(upd_str)

        # Key/Value Update (Exponential Moving Average)
        alpha = epi_write_alpha * ws
        beta  = epi_write_beta  * ws

        old_k = epi_keys[jnp.arange(B), slot_idx, :]
        old_v = epi_vals[jnp.arange(B), slot_idx, :]

        new_k = (1.0 - alpha[:, None]) * old_k + (alpha[:, None]) * write_key
        new_v = (1.0 - beta [:, None]) * old_v + (beta [:, None]) * write_val

        new_k = new_k / (jnp.linalg.norm(new_k, axis=-1, keepdims=True) + 1e-6) # Normalize

        epi_keys_new = epi_keys.at[jnp.arange(B), slot_idx, :].set(new_k)
        epi_vals_new = epi_vals.at[jnp.arange(B), slot_idx, :].set(new_v)

        return epi_keys_new, epi_vals_new, epi_age_new, epi_strength_new, slot_idx, best_sim.squeeze(-1)


# SGRM WRITER (WindowSalienceWriter)
# Санах ойд юу бичих вэ гэдгийг шийддэг хэсэг.

class WindowSalienceWriter(nn.Module):
    """
    Multi-Head Salience Writer:
    Цонхны бүх токенуудаас хамгийн чухал гэсэн хэсгүүдийг ялгаж аваад, нэгтгэж нэг вектор (Summary) болгоно.
    """
    embed_dim    : int
    num_heads    : int = 4
    dropout_rate : float = 0.0

    @nn.compact
    def __call__(self, x, mask_bool, deterministic=True):
        B, T, D = x.shape
        head_dim = D // self.num_heads
        x_heads = x.reshape(B, T, self.num_heads, head_dim)

        # Learnable Temperature (Тодруулах хүч)
        temp_param  = self.param("temp", nn.initializers.ones, (self.num_heads,))
        temperature = nn.softplus(temp_param) + 0.3

        # Анхаарлын (Salience) оноо тооцох
        salience_proj = nn.Dense(self.num_heads, name="salience_logits")(x)
        logits        = salience_proj / temperature[None, None, :]

        # Padding болон хоосон зайг хасч тооцох
        mask_expanded = mask_bool[:, :, None]
        safe_logits   = jnp.where(mask_expanded, logits, -1e9)

        # Softmax & Pooling
        max_logits = jax.lax.stop_gradient(jnp.max(safe_logits, axis=1, keepdims=True))
        exps       = jnp.exp(safe_logits - max_logits)
        exps       = jnp.where(mask_expanded, exps, 0.0)
        sum_exps   = jnp.sum(exps, axis=1, keepdims=True) + 1e-6
        weights    = exps / sum_exps  # (B, T, H)

        write_vec_heads = jnp.sum(weights[:, :, :, None] * x_heads, axis=1)

        # Write Strength Gate (Энэ цонх санах ойд бичих эрхтэй юу?)
        gate_logits   = nn.Dense(1, name="gate_proj")(write_vec_heads)
        window_valid  = jnp.any(mask_bool, axis=1)[:, None, None]
        write_u_heads = nn.sigmoid(gate_logits) * window_valid.astype(jnp.float32)

        write_vec = write_vec_heads.reshape(B, D)
        write_vec = RMSNorm(self.embed_dim)(write_vec)
        if self.dropout_rate > 0.0:
            write_vec = nn.Dropout(self.dropout_rate, deterministic=deterministic)(write_vec)

        write_strength_heads = write_u_heads.squeeze(-1)
        write_strength = jnp.mean(write_strength_heads, axis=-1, keepdims=True)

        return write_vec, write_strength, write_strength_heads


# MULTI-HEAD ENGRAM MEMORY (STATIC KNOWLEDGE), DeepSeek загварын Lookup Table санаа.

class NgramEngramMemory(nn.Module):
    vocab_size   : int
    embed_dim    : int
    memory_size  : int   = 100000
    ngram_n      : int   = 4
    num_heads    : int   = 4
    dropout_rate : float = 0.05

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, "Embed dim divisible error"
        self.head_dim = self.embed_dim // self.num_heads

        # Санах ойн том хүснэгт (Embedding Table)
        self.memory_table = self.param(
            "engram_table", nn.initializers.normal(stddev=0.02),
            (self.memory_size, self.num_heads, self.head_dim)
        )
        self.gate_logit = self.param(
            "engram_gate", nn.initializers.constant(-2.0),
            (self.num_heads, self.head_dim)
        )
        # Hashing хийх анхны тоонуудыг бэлдэх
        ps   = []
        base = 131
        for h in range(self.num_heads):
            x, row = base + h * 1009, []
            for _ in range(self.ngram_n):
                row.append(x); x = (x * 31) + 1
            ps.append(row)
        self.primes = jnp.array(ps, dtype=jnp.uint32)

    @nn.compact
    def __call__(self, current_ids, prev_ids_overlap, deterministic=True):
        # Rolling Hash тооцоолол (Vectorized)
        B, W = current_ids.shape
        O    = prev_ids_overlap.shape[1]

        current_ids      = jnp.where(current_ids == pad_id, 0, current_ids)
        prev_ids_overlap = jnp.where(prev_ids_overlap == pad_id, 0, prev_ids_overlap)
        full_seq         = jnp.concatenate([prev_ids_overlap, current_ids], axis=1).astype(jnp.uint32)
        start_idx        = O

        hash_sums = jnp.zeros((B, W, self.num_heads), dtype=jnp.uint32)
        for i in range(self.ngram_n):
            chunk     = full_seq[:, start_idx-i : full_seq.shape[1]-i]
            p_vec     = self.primes[:, i]
            hash_sums = hash_sums + (chunk[:, :, None] * p_vec[None, None, :])

        lookup_indices = (hash_sums % self.memory_size).astype(jnp.int32)

        # Table Lookup (Parallel Gather)
        table_h = jnp.transpose(self.memory_table, (1, 0, 2))
        idx_h   = jnp.transpose(lookup_indices, (2, 0, 1))
        got_h   = jax.vmap(lambda t, i: t[i], in_axes=(0, 0), out_axes=0)(table_h, idx_h)
        retrieved = jnp.transpose(got_h, (1, 2, 0, 3))

        # Gating & Output
        gate = jax.nn.sigmoid(self.gate_logit)
        out  = retrieved * gate[None, None, :, :]
        out  = out.reshape(B, W, self.embed_dim)
        out  = nn.Dropout(self.dropout_rate, deterministic=deterministic)(out)
        return out


# TRANSFORMER BLOCKS

class CausalSelfAttention(nn.Module):
    embed_dim : int
    num_heads : int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        if kv is None: kv = x
        B, Tq, _ = x.shape; _, Tk, _ = kv.shape
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
                q, k = apply_rope(q, f_q), apply_rope(k, f_k)
            else:
                q, k = apply_rope(q, freqs_cis[:, :Tq]), apply_rope(k, freqs_cis[:, :Tk])

        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        if mask is not None: attn_weights = jnp.where(mask, attn_weights, -1e9)

        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(0.1, deterministic=deterministic)(attn_probs)
        out        = jnp.matmul(attn_probs, v).transpose(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)
        return nn.Dense(self.embed_dim, name="out_proj")(out)

class TransformerBlock(nn.Module):
    embed_dim     : int
    num_heads     : int
    deterministic : bool = True  

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None):
        norm_x  = RMSNorm(self.embed_dim)(x)
        norm_kv = kv if kv is not None else RMSNorm(self.embed_dim)(x)
        
        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(
            norm_x, 
            mask          = mask, 
            kv            = norm_kv, 
            freqs_cis     = freqs_cis, 
            deterministic = self.deterministic
        )
        x = x + SwiGLU(self.embed_dim)(
            RMSNorm(self.embed_dim)(x), 
            deterministic=self.deterministic
        )
        return x

# CWRRTE RECURRENT CELL (THE CORE ENGINE)
# Цонх бүр дээр ажиллах логик, энд бүх санах ойн механизмууд нийлнэ.

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
    deterministic     : bool = True

    @nn.compact
    def __call__(self, carry, tokens_w):
        # State (Carry) задлах
        mem_emb, mem_ids, ssum_fast, ssum_slow, epi_keys, epi_vals, epi_age, epi_strength, last_write_key, pos_base = carry
        B, T = tokens_w.shape; O = self.overlap
        S    = self.window_len - self.overlap

        # Input Embedding & Injection
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)

        # Хуучин хураангуйг (ssum) оруулах (Legacy connection)
        global_ctx     = jnp.concatenate([ssum_fast, ssum_slow], axis=-1)
        ctx_proj       = RMSNorm(self.embed_dim)(nn.Dense(self.embed_dim)(global_ctx))
        ctx_bc         = jnp.broadcast_to(ctx_proj[:, None, :], x.shape)
        injection_gate = nn.sigmoid(nn.Dense(self.embed_dim)(jnp.concatenate([x, ctx_bc], axis=-1)))
        x              = x + (ctx_proj[:, None, :] * injection_gate)

        # Memory Retrievals
        # Engram Lookup
        engram_emb = NgramEngramMemory(
            self.vocab_size,
            self.embed_dim,
            self.engram_vocab_size,
            self.engram_ngram_n,
            self.engram_num_heads,
            engram_dropout
        )(tokens_w, mem_ids, deterministic=self.deterministic)
        engram_emb = RMSNorm(self.embed_dim)(engram_emb)

        mem_processed = RMSNorm(self.embed_dim)(nn.Dense(self.embed_dim)(mem_emb))

        # Episodic Slot Reading (Targeted Read)
        q_win        = jnp.mean(RMSNorm(self.embed_dim)(x), axis=1) # Window query
        writer_proxy = jnp.clip(jnp.linalg.norm(q_win, axis=-1, keepdims=True) / math.sqrt(self.embed_dim), 0.0, 1.0)
        novelty      = 1.0 - cosine_sim(jax.lax.stop_gradient(q_win), jax.lax.stop_gradient(ssum_slow))[:, None]

        # Gate & Read
        g_recall                = MemoryRecallGate(self.embed_dim)(q_win, writer_proxy, novelty, deterministic=self.deterministic)
        epi_read, _, epi_logits = EpisodicSlotReader(self.embed_dim)(q_win, epi_keys, epi_vals, epi_age, epi_strength, deterministic=self.deterministic)

        # Recall gate нь санах ойг ашиглах эсэхийг шийддэг тул уншсан үр дүнг
        # үндсэн урсгалд оруулахдаа gate-ийн тогтвортой утгыг ашиглана.
        g_recall_stop = jax.lax.stop_gradient(g_recall)
        x = x + nn.Dense(self.embed_dim)(epi_read * g_recall_stop)[:, None, :]

        # RoPE Frequency Setup (Absolute Position)
        # pos_base нь цонхны эхлэх байрлал (Absolute offset) бөгөөд overlap нь өмнөхөөс ирсэн хэсэгт таарна.
        head_dim  = self.embed_dim // self.num_heads
        freqs_all = precompute_freqs_cis(head_dim, self.max_seq_len + 256)

        mem_pos0  = pos_base - O
        cur_pos0  = pos_base

        # JAX Scan дотор Python slice (:) ашиглавал traced индекс static биш тул алдаа өгдөг.
        # Иймээс lax.dynamic_slice ашиглаж, эхлэх байрлал нь dynamic байж болно.
        def _freq_slice(start_pos, length):
            max_start = (freqs_all.shape[1] - length)
            start_pos = jnp.clip(start_pos, 0, max_start)
            return jax.lax.dynamic_slice(
                freqs_all,
                (0, start_pos, 0, 0),
                (1, length, 1, freqs_all.shape[3])
            )

        f_mem = _freq_slice(mem_pos0, O)
        f_eng = _freq_slice(cur_pos0, T)
        f_cur = _freq_slice(cur_pos0, T)

        f_kv  = jnp.concatenate([f_mem, f_eng, f_cur], axis=1)
        f_q   = f_cur

        # Masking Setup
        full_mask = jnp.concatenate(
            [
                jnp.ones((T, O), dtype=bool),
                jnp.tril(jnp.ones((T, T), dtype=bool)),
                jnp.tril(jnp.ones((T, T), dtype=bool)),
            ],
            axis=1
        )
        valid_k   = jnp.concatenate([jnp.ones((B, O), dtype=bool), (tokens_w != pad_id), (tokens_w != pad_id)], axis=1)
        mask      = full_mask[None, None, :, :] & valid_k[:, None, None, :]

        # Transformer Layers
        RematTransformerBlock = nn.remat(TransformerBlock)

        curr_x = x
        for i in range(self.num_layers):
            kv_seq = jnp.concatenate([mem_processed, engram_emb, curr_x], axis=1)

            # deterministic нь Dropout-ийн горим бөгөөд remat дотор Python bool байх ёстой.
            blk = RematTransformerBlock(
                self.embed_dim,
                self.num_heads,
                deterministic = self.deterministic,
                name          = f"b{i}"
            )

            curr_x = blk(
                curr_x,
                mask      = mask,
                kv        = kv_seq,
                freqs_cis = (f_q, f_kv)
            )

        curr_x = RMSNorm(self.embed_dim)(curr_x)

        # State Updates (Writing to Memory)
        new_mem_emb = curr_x[:, -O:, :]
        new_mem_ids = tokens_w[:, -O:]

        # SGRM Writer (Summary vector гаргах)
        write_vec, write_strength, write_strength_heads = WindowSalienceWriter(
            self.embed_dim,
            sgrm_num_heads,
            sgrm_dropout
        )(curr_x, (tokens_w != pad_id), deterministic=self.deterministic)

        # Legacy Summaries update (Fast/Slow)
        gate_fast     = nn.sigmoid(self.param("gate_fast", nn.initializers.constant(0.0), (self.embed_dim,)))
        new_ssum_fast = (ssum_fast * gate_fast) + (write_vec * (1.0 - gate_fast) * write_strength)
        lambda_eff    = jnp.clip(
            0.99 - (0.5 * (1.0 - cosine_sim(jax.lax.stop_gradient(write_vec), jax.lax.stop_gradient(ssum_slow))[:, None])),
            0.5,
            0.999
        )
        new_ssum_slow = (ssum_slow * lambda_eff) + (write_vec * (1.0 - lambda_eff) * write_strength)

        # Episodic Slot Write (Санах ойн үүрэнд бичих)
        write_key = nn.Dense(self.embed_dim)(write_vec)
        write_val = nn.Dense(self.embed_dim)(write_vec)
        write_key = write_key / (jnp.linalg.norm(write_key, axis=-1, keepdims=True) + 1e-6)

        epi_keys2, epi_vals2, epi_age2, epi_strength2, slot_idx, best_sim = EpisodicSlotWriter(
            self.embed_dim,
            epi_num_slots
        )(write_key, write_val, write_strength, epi_keys, epi_vals, epi_age, epi_strength, deterministic=self.deterministic)

        # Contrastive Recall Objective (Credit Assignment)
        # Өмнөх цонхонд бичсэн түлхүүр (last_write_key)-ийг одоогийн санах ойгоос
        # дахин олж чадах эсэхийг logits дээр суурилан тооцоолно.
        prev_valid = (jnp.linalg.norm(last_write_key, axis=-1) > 0.0).astype(jnp.float32)
        lk_n       = last_write_key / (jnp.linalg.norm(last_write_key, axis=-1, keepdims=True) + 1e-6)
        ek_n       = epi_keys / (jnp.linalg.norm(epi_keys, axis=-1, keepdims=True) + 1e-6)
        sim_prev   = jnp.sum(ek_n * lk_n[:, None, :], axis=-1)  # (B, K)
        target_idx = jnp.argmax(sim_prev, axis=-1)              # (B,)

        logp       = jax.nn.log_softmax(epi_logits, axis=-1)
        recall_nll = -logp[jnp.arange(B), target_idx] * prev_valid

        logits    = nn.Dense(self.vocab_size)(curr_x)
        aux       = (write_strength_heads, g_recall, recall_nll, slot_idx, best_sim)

        # pos_base нь цонхны алхалтын хэмжээгээр урагшилна (Stride = W - Overlap)
        new_pos_base = pos_base + S

        new_carry = (
            new_mem_emb, new_mem_ids,
            new_ssum_fast, new_ssum_slow,
            epi_keys2, epi_vals2, epi_age2, epi_strength2,
            write_key, new_pos_base
        )

        return new_carry, (logits, aux)


# MODEL WRAPPER

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
    def __call__(self, tokens_long, deterministic=True):
        B, N  = tokens_long.shape; W, O, S = self.window_len, self.overlap, self.window_len - self.overlap
        n_win = int(math.ceil((N - W) / S)) + 1 if N > W else 1

        # Текстийг цонхнуудад хуваах
        tokens_pad = jnp.pad(tokens_long, ((0, 0), (0, max(0, W + (n_win - 1) * S - N))), constant_values=pad_id)
        starts     = (jnp.arange(n_win) * S).astype(jnp.int32)
        windows    = jax.vmap(lambda s: jax.lax.dynamic_slice(tokens_pad, (0, s), (B, W)))(starts)

        # JAX Scan ашиглан recurrent гүйлт хийх
        ScanCell = nn.scan(
            CWRRTEWindowCell,
            variable_broadcast = "params",
            split_rngs         = {"params": False, "dropout": True},
            in_axes            = 0,
            out_axes           = 0
        )

        # Initial States
        # pos_base нь RoPE-д ашиглагдах абсолют байрлал бөгөөд overlap хэсэг 0..O-1 дээр сууж,
        # анхны цонхны эхлэл 0 дээр эхэлнэ.
        init_state = (
            jnp.zeros((B, O, self.embed_dim)),                                  # Overlap Emb
            jnp.zeros((B, O), dtype=jnp.int32),                                 # Overlap IDs
            jnp.zeros((B, self.embed_dim)),                                     # Fast Summary
            jnp.zeros((B, self.embed_dim)),                                     # Slow Summary
            jnp.zeros((B, epi_num_slots, self.embed_dim)),                      # Epi Keys
            jnp.zeros((B, epi_num_slots, self.embed_dim)),                      # Epi Vals
            jnp.ones ((B, epi_num_slots)) * 1e3,                                # Epi Age
            jnp.zeros((B, epi_num_slots)),                                      # Epi Strength
            jnp.zeros((B, self.embed_dim)),                                     # Last Write Key
            jnp.array(0, dtype=jnp.int32)                                       # pos_base нь цонхны эхлэх абсолют байрлал буюу Window start position юм.
        )

        _, (logits_windows, aux_windows) = ScanCell(
            vocab_size        = self.vocab_size,
            embed_dim         = self.embed_dim,
            num_layers        = self.num_layers,
            num_heads         = self.num_heads,
            window_len        = self.window_len,
            overlap           = self.overlap,
            engram_vocab_size = self.engram_vocab_size,
            engram_ngram_n    = self.engram_ngram_n,
            engram_num_heads  = self.engram_num_heads,
            max_seq_len       = self.max_seq_len,
            deterministic     = deterministic
        )(init_state, windows)

        # Гаралтыг буцааж нийлүүлэх
        out = logits_windows[0]
        if n_win > 1:
            rest = logits_windows[1:, :, O:, :].transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            out  = jnp.concatenate([out, rest], axis=1)
        return out[:, :N, :], aux_windows


#  TRAINING LOOP & LOSS FUNCTIONS

model = CWRRTETransformer(
    vocab_size,
    embed_dim,
    num_layers,
    num_heads,
    cwr_window_len,
    cwr_overlap,
    engram_vocab_size,
    engram_ngram_n,
    engram_num_heads,
    max_seq_len
)

@jax.jit
def train_step(state, batch, rng):
    """
    Нэг сургалтын алхам.
    Нийт алдаа (Total Loss) = Text Loss + Write Budget + Recall Budget + Contrastive Recall
    """
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(p):
        logits, aux_windows = model.apply({"params": p}, batch[:, :-1], deterministic=False, rngs={"dropout": dropout_rng})
        labels = batch[:, 1:]
        logits = logits[:, :labels.shape[1], :]

        # Үндсэн Text Generation Loss
        loss_t    = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask      = (labels != pad_id).astype(jnp.float32)
        text_loss = jnp.sum(loss_t * mask) / (jnp.sum(mask) + 1e-6)

        # Auxiliary Losses (Туслах loss)
        write_str_w, g_recall_w, recall_nll_w, _, _ = aux_windows

        # Write Budget  : Санах ойд хэт их бичихээс сэргийлэх
        write_budget_loss  = (jnp.mean(write_str_w) - sgrm_target_rate) ** 2
        # Recall Budget : Санах ойг хэт их уншихаас сэргийлэх
        recall_budget_loss = (jnp.mean(g_recall_w) - recall_target_rate) ** 2
        # Contrastive   : Санах ойн бүрэн бүтэн байдал
        contrastive_loss   = jnp.mean(recall_nll_w)

        total_loss = text_loss + \
            (sgrm_budget_weight   * write_budget_loss ) + \
            (recall_budget_weight * recall_budget_loss) + \
            (contrastive_weight   * contrastive_loss  )
        return total_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_rng

@jax.jit
def predict_step_jit(params, fixed_input):
    return model.apply({"params": params}, fixed_input, deterministic=True)[0]

def generate(params, prompt, gen_len=100, temp=0.8):
    token_ids = list(encode_text(prompt))
    if token_ids[-1] == eos_id: token_ids.pop()
    print(f"Текст үүсгэж байна: '{prompt}'")

    for _ in range(gen_len):
        if len(token_ids) >= sft_long_seq_len: break
        inp_np = np.array(token_ids + [pad_id] * (sft_long_seq_len - len(token_ids)), dtype=np.int32)
        logits = predict_step_jit(params, jnp.array([inp_np]))[0, len(token_ids) - 1, :]

        logits = logits.at[pad_id].set(-1e9).at[bos_id].set(-1e9)
        probs  = np.exp(np.array(logits) / temp)
        probs /= np.sum(probs) if np.sum(probs) > 0 else 1

        next_id = np.random.choice(len(probs), p=probs)
        token_ids.append(next_id)
        if next_id == eos_id: break

    return decode_ids(token_ids)

# DEBUG FUNCTIONS
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


#  MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps"  , type=int, default=sft_total_steps )
    parser.add_argument("--seq-len", type=int, default=sft_long_seq_len)
    parser.add_argument("--batch"  , type=int, default=sft_batch_size  )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  CWRRTES : Cross-Window Recurrent Transformer + Episodic Slot Memory")
    print("  Features: Targeted Read/Write + Recall Gate + Integrity + Contrastive Recall")
    print(f"  Steps: {args.steps} | Batch: {args.batch} | SeqLen: {args.seq_len}")
    print(f"  Engram Size: {engram_vocab_size} | EpiSlots: {epi_num_slots}")
    print("="*70 + "\n")

    rng = jax.random.PRNGKey(seed); rng, init_rng = jax.random.split(rng)
    dummy_in  = jnp.zeros((1, args.seq_len), dtype=jnp.int32)
    params    = model.init(init_rng, dummy_in, deterministic=True)["params"]
    print(f"Моделийн нийт параметр: {sum(x.size for x in jax.tree_util.tree_leaves(params))/1e6:.2f}M")

    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(1e-6, sft_learning_rate, sft_warmup_steps, args.steps, 1e-6),
        weight_decay=weight_decay
    ))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    start_time = time.time()
    for step in range(1, args.steps + 1):
        starts           = np.random.randint(0, len(corpus_ids) - args.seq_len - 1, args.batch)
        batch_jax        = jnp.asarray(np.stack([corpus_ids[s : s + args.seq_len + 1] for s in starts]), dtype=jnp.int32)
        state, loss, rng = train_step(state, batch_jax, rng)

        if step % sft_loss_freq == 0:
            dt = time.time() - start_time
            est, _ = _find_engram_subtree(state.params)
            gate_info = f"| EG_Mean: {_sigmoid_gate_stats(est)[0]:.3f}" if est else ""
            print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s {gate_info}")

        if step % sft_sample_freq == 0:
            print("\n" + "-"*40)
            sample_text = generate(state.params, sample_prompt_text, gen_len=sample_gen_len, temp=sample_temp)
            print(f"ГАРСАН ҮР ДҮН: {sample_text}")
            speak_async(sample_text, voice=tts_voice, speed=tts_speed, amp=tts_amp, enabled=tts_enabled)
            print("-"*40 + "\n")

    print("Сургалт амжилттай дууслаа!")

if __name__ == "__main__":
    main()