#
#  CWRRTESOUP V3 : Урт текст боловсруулах шатлалтай санах ойт transformer
#
#  АРХИТЕКТУРЫН ДЭЛГЭРЭНГҮЙ ТАЙЛБАР
#
#    Урт текстийг олон цонх (window) болгон тасалж боловсруулах үедээ
#    өмнөх цонхнуудаас хэрэгтэй мэдээллээ алдалгүйгээр хадгалж, шаардлагатай үедээ
#    буцаан татаж (recall) ашиглах явдал юм.
#
#
#  Үндсэн санаа нь:
#    - Дэд агуулга (window) дээр ажилладаг стандарт transformer урсгал дээр
#      нэмэлт санах ойн модулиудыг давхар суулгаж өгнө.
#    - Санах ой нь шатлалтай бүтэцтэй:
#       - L1 (Episodic Slots): Хурдан, жижиг, ойрхон санах ой (cache)
#       - L2 (Cold Bank     ): Том, удаан хугацааны хадгалалт (ring buffer)
#
#
#  Загварн доторхи бүрэлдэхүүн хэсгүүд:
#
#  1. Контекст-хаалгатай Engram санах ой (N-gram hashing)
#    - Давтагддаг тогтмол хэллэгүүдийг (n-gram) attention-аар тооцохын оронд
#      хэш хүснэгтээс дээрхи embedding vector-оос нь шууд татаж ашиглана.
#    - Char-level tokenizer ашиглаж байгаа тул ngram_n=8 гэх мэт урт n-gram ашиглаж болно.
#    - Гарч ирсэн engram embedding-ийг шууд residual урсгалд нийлүүлнэ (inject).
#    - Ингэснээр KV cache урт нэмэгдүүлэхгүй, цонхны attention тооцоолол хөнгөрнө.
#
#  2. Multi-Vector Salience Writer (SGRM)
#    - Цонхон доторх бүх токенуудын дундаас чухал мэдээллийг K ширхэг вектор болгон шахаж гаргана 
#      (K = write_heads).
#    - Энэ нь нэг цонхноос олон event-үүд гаргаж бичих боломж олгоно.
#    - Эдгээр бичих вектор бүрт бичих хүч буюу write strength, gate гэх мэтийг тооцно.
#
#  3. Hierarchical Memory Manager (L1 + L2)
#    - READ:
#        - Одоогийн цонхны query (q_win)-аар L1 слотуудыг оноо өгч уншина.
#        - L2 банкнаас top-k төстэй векторуудыг авч жигнэж уншина.
#        - L1 + L2-г нэгтгээд (merge) цэвэршүүлж (norm) цонх руу буцаана.
#    - WRITE:
#        - Multi-writer-аас ирсэн K векторуудаас хамгийн хүчтэйг L1-д бичих гол бичигч болгон сонгоно.
#        - L1-д бичихдээ:
#            - Хэрэв хамгийн төстэй слот хангалттай ойр бол (threshold) шинэчлэх (update)
#            - Үгүй бол eviction score-оор хамгийн муу слотыг сонгон дарж бичих (evict)
#        - L2-д K вектор бүгдийг ring buffer маягаар ээлжлэн хадгална (batch-safe).
#
#  4. Recall Gate + Transformer Block
#    - Recall Gate:
#        - Одоогийн контекст шинэ бөгөөд ойлгомжгүй байх үед санах ойг ашиглах хэрэгцээ өсдөг.
#        - novelty + writer_strength + q_win дээр үндэслэн recall-ийн gate тооцно.
#    - Transformer:
#        - KV = [өмнөх overlap хэсгийн mem_pr] + [одоогийн цонхны hidden]
#        - RoPE-г global байрлал (absolute pos) дээр тааруулж, цонх дамжсан ч зөв эргүүлнэ.
#        - Маск нь causal + valid token гэсэн 2 нөхцөлийг зэрэг хангана.
#
#  5. Window Cell
#    - Нэг цонхыгг боловсруулах эс буюу cell.
#    - Carry-д:
#        mem_emb/mem_ids (overlap cache), ssum_fast/ssum_slow (summary),
#        L1 slots (keys/vals/age/strength), L2 bank/ptr, last_key, pos_base
#    - Cell:
#        - Input embedding -> Legacy summary injection
#        - Engram injection
#        - SGRM writer-аар K write vecs үүсгэх
#        - Memory manager-аар read_out авах + L1/L2 update хийх
#        - Recall gate-ээр read_out-г нэмэх
#        - Transformer layer-ууд
#        - Summary update
#        - Contrastive маягийн nll target гаргаж aux-д өгнө
#
#  6. Үндсэн transformer wrapper (JAX scan ашиглана)
#    - Урт дарааллыг олон цонх болгон vmap + scan ашиглан боловсруулна.
#    - params-ийг scan дотор broadcast хийж, dropout RNG-г тусад нь салгана.
#
#
#  Loss-ийн бүрэлдэхүүнүүд:
#    1. Text cross-entropy (үндсэн хэлний загварчлалд ашиглана)
#    2. Writer budget (SGRM gate-ийн дундаж идэвхийг target rate руу барих)
#    3. Recall budget (Recall gate-ийн дундаж идэвхийг target rate руу барих)
#    4. Contrastive маягийн санах ойн зөв уншилтын nll (санах ойн бүрэн бүтэн байдалд хэрэгтэй)
#
#
#  Гол практик ач холбогдлууд нь:
#    - Цонх дамжсан урт контекст дээр тогтвортой ажиллах
#    - Санах ойг хэт их ухахгүй, зөвхөн хэрэгтэй үед нь авч ашиглах
#    - Давтагддаг мэдээллийг хямд аргаар (hash table) татах
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


# Hyperparameters

# Өгөгдөл
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# CWRRTE Архитектур (Window)
cwr_window_len        = 128
cwr_overlap           = 32

# Engram Санах ой (Char-level тул N=8)
engram_vocab_size     = 100000
engram_ngram_n        = 8
engram_num_heads      = 4
engram_dropout        = 0.05

# SGRM (Multi-Vector Write)
sgrm_dropout          = 0.00
sgrm_num_heads        = 4
sgrm_write_heads      = 4      # Нэг цонхноос K ширхэг бичих вектор гаргана
sgrm_target_rate      = 0.20
sgrm_budget_weight    = 0.1

# Episodic Memory (L1 Cache)
epi_num_slots         = 64
epi_strength_decay    = 0.995
epi_age_penalty       = 0.02
epi_strength_boost    = 0.50
epi_write_alpha       = 0.50
epi_min_strength      = 1e-3

# L1 Eviction Policy
epi_evict_beta_str    = 5.0
epi_evict_gamma_age   = 0.1
epi_update_threshold  = 0.85

# L2 Memory Bank (Cold Storage)
l2_memory_size        = 2048
l2_read_topk          = 8
l2_read_temp          = 1.0

# Recall & Loss
recall_target_rate    = 0.15
recall_budget_weight  = 0.05
contrastive_weight    = 0.05

# Training
sft_total_steps       = 5000
sft_long_seq_len      = 1024
sft_batch_size        = 8
sft_learning_rate     = 5e-4
sft_warmup_steps      = 100
sft_loss_freq         = 10
sft_sample_freq       = 100

# TTS & Generation
sample_gen_len        = 256
sample_temp           = 0.8
sample_prompt_text    = "Once upon a time there was a little robot. "
tts_enabled           = True
tts_voice             = "en"; tts_speed = 165; tts_amp = 120; tts_max_chars = 400

# Model Size
num_layers            = 6
num_heads             = 8
embed_dim             = 512
max_seq_len           = 8192
max_grad_norm         = 1.0
weight_decay          = 0.01

np.random.seed(seed)
random.seed(seed)


# ESPEAK TTS

_ESPEAK_BIN = shutil.which("espeak-ng") or shutil.which("espeak")

def _tts_clean(s, max_chars=400):
    # TTS-д өгөгдөх текстийг цэвэрлэх (newline, илүүдэл зай, хэвлэгддэггүй тэмдэг)
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = "".join(ch for ch in s if ch.isprintable())
    return s[:max_chars]

def speak_async(text, voice="en", speed=165, amp=120, enabled=True):
    # Non-blocking TTS (Сургалт/генерацид саад болохгүй)
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


# Өгөгдөл бэлтгэх

print(">>> Өгөгдлийг уншиж байна...")

if not os.path.exists(dataset_path):
    print("Анхаар: Dataset олдсонгүй. Хиймэл текст үүсгэж байна.")
    dummy_vocab = ["robot", "girl", "boy", "dog", "cat", "run", "jump", "happy", "sad", "the", "a", "is", "was"]
    raw_text    = ""
    for _ in range(2000):
        sent = " ".join(random.choices(dummy_vocab, k=10)) + ". "
        raw_text += sent + end_of_text_token
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

# Character-level tokenizer
all_stories   = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 10]
unique_chars  = sorted(list(set("".join(all_stories))))

PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"
chars         = [PAD, BOS, EOS] + unique_chars
char_to_id    = {c: i for i, c in enumerate(chars)}
id_to_char    = {i: c for c, i in char_to_id.items()}
vocab_size    = len(chars)

pad_id, bos_id, eos_id = char_to_id[PAD], char_to_id[BOS], char_to_id[EOS]

def encode_text(text):
    # Текстийг ID дараалал болгон хувиргах
    return [bos_id] + [char_to_id.get(ch, pad_id) for ch in text] + [eos_id]

def decode_ids(ids):
    # ID-уудыг буцааж текст болгох (PAD/BOS/EOS-ийг алгасана)
    return "".join([id_to_char[int(i)] for i in ids if int(i) not in [pad_id, bos_id, eos_id]])

flat_tokens = []
for s in all_stories:
    flat_tokens.extend(encode_text(s))

corpus_ids = np.array(flat_tokens, dtype=np.int32)
print(f"Dataset Stats: Vocab={vocab_size}, Total Tokens={len(corpus_ids)}")


# Туслах функцууд

def apply_rope(x, freq_cis):
    """
    RoPE (Rotary Positional Embedding)
    - (even, odd) комплекс утгаар эргүүлж байрлалын мэдээлэл бий болгоно.
    """
    B, T, H, D = x.shape
    x_complex  = jax.lax.complex(x[..., 0::2], x[..., 1::2])
    x_rotated  = x_complex * freq_cis
    x_out      = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1).reshape(B, T, H, D)
    return x_out

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    """
    RoPE-ийн global давтамжийн cache
    - absolute position ашиглах тул цонх дамжих үед ч зөв зүсэж авах боломжтой.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t     = jnp.arange(max_len)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)[None, :, None, :]

def slice_freqs_cis(freqs_cis_global, pos_idx):
    # Global RoPE кэшээс тухайн absolute position-уудын хэсгийг зүсэж авах
    return jnp.take(freqs_cis_global, pos_idx.astype(jnp.int32), axis=1)

def cosine_sim(a, b):
    # Cosine similarity (векторуудын чиглэлийн ойролцоо байдлыг хэмжинэ)
    a_n = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
    b_n = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-6)
    return jnp.sum(a_n * b_n, axis=-1)


# Суурь NN давхаргууд

class RMSNorm(nn.Module):
    """
    RMSNorm
    - LayerNorm-оос хөнгөн, тогтвортой нормализац.
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
    SwiGLU (MLP)
    - Gate механизмаар мэдээлэл шүүн дамжуулна.
    """
    embed_dim: int
    expand_factor: int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden_dim = int(self.embed_dim * self.expand_factor)
        gate_val   = nn.Dense(hidden_dim * 2, use_bias=False)(x)
        gate, val  = jnp.split(gate_val, 2, axis=-1)
        out        = nn.Dense(self.embed_dim, use_bias=False)(nn.silu(gate) * val)
        return nn.Dropout(0.1, deterministic=deterministic)(out)


# 1. CONTEXT-GATED ENGRAM MEMORY

class NgramEngramMemory(nn.Module):
    """
    Engram Memory (N-gram Hash Lookup)
    - prev_ids + curr_ids дээр rolling hash тооцоолж, хүснэгтээс embedding татна.
    - x_input (одоогийн hidden/embedding) дээр суурилсан gate-ээр нөлөөг нь тохируулна.
    """
    vocab_size  : int
    embed_dim   : int
    memory_size : int
    ngram_n     : int
    num_heads   : int

    def setup(self):
        # Толгой бүрийн хэмжээс
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        # Engram хүснэгт (Hash table)
        self.table = self.param(
            "engram_table",
            nn.initializers.normal(0.02),
            (self.memory_size, self.num_heads, self.head_dim)
        )

        # Gate тооцоолох projection (x_input болон engram raw-г нийлүүлээд sigmoid)
        self.gate_proj = nn.Dense(self.embed_dim)

        # Rolling hash-д ашиглах prime/weight матриц
        ps   = []
        base = 131
        for h in range(self.num_heads):
            x = base + h * 1009
            r = []
            for _ in range(self.ngram_n):
                r.append(x)
                # 32-бит хүрээнд барьж integer overflow-оос зайлсхийх
                x = ((x * 31) + 1) & 0xFFFFFFFF
            ps.append(r)
        self.primes = jnp.array(ps, dtype=jnp.uint32)

    @nn.compact
    def __call__(self, x_input, curr_ids, prev_ids, deterministic=True):
        """
        x_input : (B, W, D)  - тухайн цонхны оролтын embedding/hidden
        curr_ids: (B, W)     - одоогийн цонхны token ids
        prev_ids: (B, O)     - өмнөх overlap хэсгийн token ids
        """
        B, W = curr_ids.shape
        O    = prev_ids.shape[1]

        # PAD-ийг 0 болгож, hashing-д тогтвортой болгох
        seq = jnp.concatenate(
            [
                jnp.where(prev_ids == pad_id, 0, prev_ids),
                jnp.where(curr_ids == pad_id, 0, curr_ids),
            ],
            axis=1
        ).astype(jnp.uint32)

        # Rolling hash хийхэд зүүн талын padding хэрэгтэй (ngram_n - 1)
        pad_left = self.ngram_n - 1
        seqp     = jnp.pad(seq, ((0, 0), (pad_left, 0)), constant_values=0)

        # Hash цонхны эхлэл (prev overlap дууссан байрлал)
        base = pad_left + O

        # Толгой бүрт hash цуглуулах
        h_sum = jnp.zeros((B, W, self.num_heads), dtype=jnp.uint32)
        for i in range(self.ngram_n):
            toks  = seqp[:, base - i : base - i + W]                                      # (B, W)
            h_sum = h_sum + toks[:, :, None] * self.primes[None, None, :, i]

        # Хүснэгтийн индекс
        idx = (h_sum % self.memory_size).astype(jnp.int32)                                # (B, W, H)

        # Table lookup (head-major болгож vmap хийх)
        tab_h = self.table.transpose(1, 0, 2)                                             # (H, M, hd)
        idx_h = idx.transpose(2, 0, 1)                                                    # (H, B, W)

        got     = jax.vmap(lambda t, i: t[i], in_axes=(0, 0), out_axes=0)(tab_h, idx_h)   # (H, B, W, hd)
        out_raw = got.transpose(1, 2, 0, 3).reshape(B, W, self.embed_dim)                 # (B, W, D)

        # Контекст-хаалга: x_input ба out_raw-г нийлүүлж gate тооцоолох
        combined = jnp.concatenate([x_input, out_raw], axis=-1)
        gate     = jax.nn.sigmoid(self.gate_proj(combined))

        out = out_raw * gate
        out = nn.Dropout(engram_dropout, deterministic=deterministic)(out)
        return out


# 2. MULTI-VECTOR SALIENCE WRITER

class WindowSalienceWriter(nn.Module):
    """
    Multi-Vector Writer
    - Цонхны токенууд дээр softmax жин гаргаж K ширхэг summary vector үүсгэнэ.
    - Тус бүрт нь бичих хүч (gate) тооцно.
    """
    embed_dim   : int
    num_heads   : int = 4
    write_heads : int = 4

    @nn.compact
    def __call__(self, x, mask_bool, deterministic=True):
        B, T, D = x.shape
        K       = self.write_heads

        # Token -> K оноо (score)
        scores = nn.Dense(K)(x)                            # (B, T, K)

        # PAD байрлалыг тооцоололд оруулахгүй
        mask_bc = mask_bool[:, :, None]                    # (B, T, 1)
        scores  = jnp.where(mask_bc, scores, -1e9)

        # Token-уудын дагуу нормчилол хийж K жин гаргана
        weights = jax.nn.softmax(scores, axis=1)           # (B, T, K)

        # K ширхэг вектор (events) гаргах
        vecs = jnp.einsum("btk,btd->bkd", weights, x)      # (B, K, D)
        vecs = RMSNorm(self.embed_dim)(vecs)

        # Вектор бүрийн бичих хүч (0..1)
        gate_logits    = nn.Dense(1)(vecs)                 # (B, K, 1)
        write_strength = nn.sigmoid(gate_logits)           # (B, K, 1)

        return vecs, write_strength


# 3. HIERARCHICAL MEMORY MANAGER

class HierarchicalMemoryManager(nn.Module):
    """
    Шатлалтай санах ой (L1 + L2)
    - READ  : L1 slots + L2 bank уншаад нэгтгэнэ
    - WRITE : L1-д 1 гол вектор бичнэ, L2-д K вектор бүгдийг ring buffer-д бичнэ
    """
    embed_dim : int
    num_slots : int
    l2_size   : int
    l2_topk   : int   = 8
    l2_temp   : float = 1.0

    @nn.compact
    def __call__(
        self,
        curr_query,
        write_vecs,
        write_gates,
        l1_keys,
        l1_vals,
        l1_age,
        l1_str,
        l2_bank,
        l2_ptr,
        deterministic=True
    ):
        B, K, D = write_vecs.shape
        S       = self.num_slots
        L2      = self.l2_size


        # READ: L1 (Episodic Slots)

        qn  = curr_query / (jnp.linalg.norm(curr_query, axis=-1, keepdims=True) + 1e-6)
        k1n = l1_keys    / (jnp.linalg.norm(l1_keys   , axis=-1, keepdims=True) + 1e-6)

        sim_l1 = jnp.sum(k1n * qn[:, None, :], axis=-1)  # (B, S)

        logits_l1 = sim_l1 \
            + (epi_strength_boost * jnp.log(jnp.clip(l1_str, 1e-3, 1e9))) \
            - (epi_age_penalty * l1_age)

        # Хоосон/сул слотуудыг унших магадлалыг маш бага болгох
        logits_l1 = logits_l1 + ((l1_str > 1e-3).astype(jnp.float32) - 1.0) * 1e3

        w_l1    = jax.nn.softmax(logits_l1, axis=-1)
        read_l1 = jnp.sum(w_l1[:, :, None] * l1_vals, axis=1)                            # (B, D)


        # READ: L2 (Cold Bank) top-k

        k2n   = l2_bank / (jnp.linalg.norm(l2_bank, axis=-1, keepdims=True) + 1e-6)
        sim_l2 = jnp.sum(k2n * qn[:, None, :], axis=-1)                                  # (B, L2)

        topv, topi = jax.lax.top_k(sim_l2, k=min(self.l2_topk, L2))                      # (B, k)
        l2_vecs    = jnp.take_along_axis(l2_bank, topi[:, :, None], axis=1)
        l2_w       = jax.nn.softmax(topv / self.l2_temp, axis=-1)
        read_l2    = jnp.sum(l2_w[:, :, None] * l2_vecs, axis=1)                         # (B, D)

        # L1 + L2 нэгтгэх
        read_out = read_l1 + nn.Dense(self.embed_dim, name="l2_merge")(read_l2)
        read_out = RMSNorm(self.embed_dim)(read_out)


        # WRITE: L1 (гол бичигч сонгох)

        gate_k = write_gates.squeeze(-1)                                                 # (B, K)
        best_k = jnp.argmax(gate_k, axis=-1)                                             # (B,)

        w_vec  = write_vecs[jnp.arange(B), best_k, :]                                    # (B, D)
        w_str  = write_gates[jnp.arange(B), best_k, :]                                   # (B, 1)
        w_key  = w_vec / (jnp.linalg.norm(w_vec, axis=-1, keepdims=True) + 1e-6)

        # L1 дотор update эсвэл eviction зорилго сонгох
        sim_w     = jnp.sum(k1n * w_key[:, None, :], axis=-1)                            # (B, S)
        max_sim   = jnp.max(sim_w, axis=-1)                                              # (B,)
        update_idx = jnp.argmax(sim_w, axis=-1)                                          # (B,)

        # Eviction score: age өндөр, strength бага байх тусам eviction буюу гаргах магадлал өснө
        evict_score = (0.10 * sim_w) + (epi_evict_gamma_age * l1_age) - (epi_evict_beta_str * l1_str)
        evict_idx   = jnp.argmax(evict_score, axis=-1)

        is_update   = (max_sim > epi_update_threshold)
        target_idx  = jnp.where(is_update, update_idx, evict_idx).astype(jnp.int32)

        target_mask = jax.nn.one_hot(target_idx, S, dtype=jnp.float32)                   # (B, S)
        m           = target_mask[:, :, None]                                            # (B, S, 1)

        # Update бол арай хурдан бичиж, шинэ слот бол ердийн alpha хэрэглэнэ
        alpha_val = jnp.where(is_update, 0.50, epi_write_alpha).astype(jnp.float32)
        alpha     = (alpha_val[:, None] * w_str).astype(jnp.float32)                     # (B, 1)
        a         = alpha[:, None, :]                                                    # (B, 1, 1) -> broadcast

        # Key/Value шинэчлэх (interpolation)
        l1_keys_new = l1_keys * (1.0 - a * m) + (a * m) * w_key[:, None, :]
        l1_vals_new = l1_vals * (1.0 - a * m) + (a * m) * w_vec[:, None, :]
        l1_keys_new = l1_keys_new / (jnp.linalg.norm(l1_keys_new, axis=-1, keepdims=True) + 1e-6)

        # Age: бүх слот хуучирна, харин сонгогдсон слот 0 руу ойртон reset маягтай болно
        l1_age_new = (l1_age + 1.0) * (1.0 - target_mask)

        # Strength: decay + сонгогдсон слот дээр нэмэлтээр өсгөх
        l1_str_new = l1_str * epi_strength_decay + (target_mask * w_str.squeeze(-1)[:, None]) * (1.0 - l1_str * epi_strength_decay)
        l1_str_new = jnp.clip(l1_str_new, epi_min_strength, 1.0)


        # WRITE: L2 (Ring Buffer)

        idx2 = (l2_ptr[:, None] + jnp.arange(K)[None, :]) % L2

        def _write_one(bank_b, idx_b, vec_b):
            return bank_b.at[idx_b, :].set(vec_b)

        l2_bank_new = jax.vmap(_write_one)(l2_bank, idx2, write_vecs)
        l2_ptr_new  = (l2_ptr + K) % L2

        dbg = {"logits_l1": logits_l1}
        new_state = (l1_keys_new, l1_vals_new, l1_age_new, l1_str_new, l2_bank_new, l2_ptr_new)

        return read_out, new_state, dbg


# 4. RECALL & TRANSFORMER

class MemoryRecallGate(nn.Module):
    """
    Recall Gate
    - Одоогийн цонх дээр санах ой хэрэгтэй эсэхийг (0..1) шийднэ.
    - novelty буюу шинэ сэдэв шинэ мэдээлэл ихсэх тусам санах ойг ухах магадлал өснө.
    """
    embed_dim: int

    @nn.compact
    def __call__(self, q_win, writer_strength, novelty):
        x = jnp.concatenate([q_win, writer_strength, novelty], axis=-1)
        h = nn.Dense(self.embed_dim)(x)
        h = nn.silu(h)
        g = nn.sigmoid(nn.Dense(1)(h))
        return g

class TransformerBlock(nn.Module):
    """
    Стандарт Transformer Block
    - KV дээр overlap + current window-г зэрэг ашиглагдана.
    - RoPE-г global position дээр тааруулна.
    """
    embed_dim     : int
    num_heads     : int
    deterministic : bool = True

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None):
        B, T, _ = x.shape
        H       = self.num_heads
        D       = self.embed_dim // H

        def attn(x_in, kv_in):
            q = nn.Dense(self.embed_dim)(x_in ).reshape(B, T, H, D)
            k = nn.Dense(self.embed_dim)(kv_in).reshape(B, kv_in.shape[1], H, D)
            v = nn.Dense(self.embed_dim)(kv_in).reshape(B, kv_in.shape[1], H, D)

            # RoPE: q болон k дээр байршлын эргэлт хийх
            if freqs_cis is not None:
                if isinstance(freqs_cis, tuple):
                    q = apply_rope(q, freqs_cis[0][:, :T])
                    k = apply_rope(k, freqs_cis[1][:, :kv_in.shape[1]])
                else:
                    q = apply_rope(q, freqs_cis[:, :T])
                    k = apply_rope(k, freqs_cis[:, :kv_in.shape[1]])

            # Attention logits
            w = jnp.matmul(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 3, 1)
            ) / math.sqrt(D)

            # Mask: causal + valid token
            if mask is not None:
                w = jnp.where(mask, w, -1e9)

            p = nn.Dropout(0.1, deterministic=self.deterministic)(jax.nn.softmax(w, axis=-1))

            out = jnp.matmul(p, v.transpose(0, 2, 1, 3))
            out = out.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
            return out

        # Pre-norm
        norm_x  = RMSNorm(self.embed_dim)(x)
        norm_kv = RMSNorm(self.embed_dim)(kv) if kv is not None else RMSNorm(self.embed_dim)(x)

        # Attention residual
        x = x + nn.Dense(self.embed_dim)(attn(norm_x, norm_kv))

        # MLP residual
        x = x + SwiGLU(self.embed_dim)(RMSNorm(self.embed_dim)(x), deterministic=self.deterministic)

        return x


# 5. WINDOW CELL

class CWRRTEWindowCell(nn.Module):
    """
    Window Cell
    - Нэг цонхны тооцоог хийж, шинэ carry буцаана.
    """
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
    l2_memory_size    : int  = 2048

    @nn.compact
    def __call__(self, carry, tokens_w, win_start, freqs_cis_global):
        # Carry задлах
        mem_emb, mem_ids, ssum_fast, ssum_slow, \
        l1k, l1v, l1a, l1s, l2b, l2p, \
        last_key, pos_base = carry

        B, T = tokens_w.shape
        O    = self.overlap

        # Input Embedding
        x_emb = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)

        # Legacy Summary Injection
        g_ctx = RMSNorm(self.embed_dim)(
            nn.Dense(self.embed_dim)(jnp.concatenate([ssum_fast, ssum_slow], axis=-1))
        )
        gate = nn.sigmoid(
            nn.Dense(self.embed_dim)(
                jnp.concatenate([x_emb, jnp.broadcast_to(g_ctx[:, None], x_emb.shape)], axis=-1)
            )
        )
        x = x_emb + g_ctx[:, None] * gate

        # Engram Injection (Hash lookup)
        engram = NgramEngramMemory(
            vocab_size  = self.vocab_size,
            embed_dim   = self.embed_dim,
            memory_size = self.engram_vocab_size,
            ngram_n     = self.engram_ngram_n,
            num_heads   = self.engram_num_heads
        )(x_emb, tokens_w, mem_ids, deterministic=self.deterministic)

        engram = RMSNorm(self.embed_dim)(engram)
        x      = x + nn.Dense(self.embed_dim, name="engram_inj")(engram)

        # Memory Writer (SGRM)
        q_win = jnp.mean(RMSNorm(self.embed_dim)(x), axis=1)

        w_vecs, w_strs = WindowSalienceWriter(
            self.embed_dim, sgrm_num_heads, sgrm_write_heads
        )(
            RMSNorm(self.embed_dim)(x),
            tokens_w != pad_id,
            deterministic=self.deterministic
        )

        # Memory Manager (L1 + L2)
        read_out, (nk, nv, na, ns, nb, np_ptr), dbg = HierarchicalMemoryManager(
            self.embed_dim,
            epi_num_slots,
            self.l2_memory_size,
            l2_read_topk,
            l2_read_temp
        )(
            q_win, w_vecs, w_strs,
            l1k, l1v, l1a, l1s,
            l2b, l2p,
            deterministic=self.deterministic
        )

        # Recall Gate (хэзээ санах вэ?)
        novelty = 1.0 - cosine_sim(jax.lax.stop_gradient(q_win), jax.lax.stop_gradient(ssum_slow))[:, None]
        g_rec   = MemoryRecallGate(self.embed_dim)(q_win, jnp.mean(w_strs, axis=1), novelty)

        x = x + nn.Dense(self.embed_dim)(read_out * g_rec)[:, None]

        # Overlap Memory KV (өмнөх O токены embedding)
        mem_pr = RMSNorm(self.embed_dim)(nn.Dense(self.embed_dim)(mem_emb))

        # RoPE (absolute position)
        rope_cur = jnp.maximum(0, pos_base + jnp.arange(T))
        rope_mem = jnp.maximum(0, pos_base - O + jnp.arange(O))

        f_cur = slice_freqs_cis(freqs_cis_global, rope_cur)
        f_mem = slice_freqs_cis(freqs_cis_global, rope_mem)
        f_kv  = jnp.concatenate([f_mem, f_cur], axis=1)

        # Attention Mask (causal + valid)
        causal = jnp.concatenate(
            [
                jnp.ones((T, O), dtype=jnp.bool_),
                jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            ],
            axis=1
        )  # (T, KV)

        valid_kv = jnp.concatenate(
            [
                jnp.ones((B, O), dtype=jnp.bool_),
                (tokens_w != pad_id)
            ],
            axis=1
        ).astype(jnp.bool_)  # (B, KV)

        mask = causal[None, None, :, :] & valid_kv[:, None, None, :]   # (B, 1, T, KV)

        # Transformer давхаргууд дамжуулна (overlap + current window зэрэг ашиглагдана)
        curr = x
        for i in range(self.num_layers):
            kv   = jnp.concatenate([mem_pr, curr], axis=1)
            curr = nn.remat(TransformerBlock)(
                self.embed_dim,
                self.num_heads,
                deterministic = self.deterministic,
                name          = f"b{i}"
            )(
                curr,
                mask      = mask,
                kv        = kv,
                freqs_cis = (f_cur, f_kv)
            )

        curr = RMSNorm(self.embed_dim)(curr)

        # Summary Updates (Fast/Slow)
        w_vec_avg = jnp.mean(w_vecs, axis=1)
        w_str_avg = jnp.mean(w_strs, axis=1)

        g_fast = nn.sigmoid(self.param("gf", nn.initializers.zeros, (self.embed_dim,)))
        sf2    = ssum_fast * g_fast + w_vec_avg * (1 - g_fast) * w_str_avg

        lam = jnp.clip(
            0.99 - 0.5 * (1.0 - cosine_sim(jax.lax.stop_gradient(w_vec_avg), jax.lax.stop_gradient(ssum_slow))[:, None]),
            0.5,
            0.999
        )
        ss2 = ssum_slow * lam + w_vec_avg * (1 - lam) * w_str_avg

        # Contrastive маягийн NLL target
        last_key_norm = jnp.linalg.norm(last_key, axis=-1, keepdims=True) + 1e-6
        prev_target   = jnp.argmax(
            jnp.sum(l1k * (last_key / last_key_norm)[:, None], axis=-1),
            axis=-1
        )
        valid_prev = (jnp.linalg.norm(last_key, axis=-1) > 1e-3).astype(jnp.float32)

        nll = -jax.nn.log_softmax(dbg["logits_l1"], axis=-1)[jnp.arange(B), prev_target] * valid_prev

        # Дараагийн алхамд last_key-г шинэчлэх (хамгийн хүчтэй writer-аас)
        best_w_idx    = jnp.argmax(w_strs.squeeze(-1), axis=-1)
        new_last_key  = w_vecs[jnp.arange(B), best_w_idx, :]

        # Carry & Output
        new_carry = (
            curr[:, -O:],          # overlap mem_emb
            tokens_w[:, -O:],      # overlap mem_ids
            sf2,
            ss2,
            nk, nv, na, ns,        # L1 шинэ төлөв
            nb, np_ptr,            # L2 шинэ төлөв
            new_last_key,
            pos_base + (self.window_len - O)
        )

        logits = nn.Dense(self.vocab_size)(curr)
        aux    = (jnp.mean(w_strs), jnp.mean(g_rec), jnp.mean(nll))

        return new_carry, (logits, aux)


# 6. MAIN TRANSFORMER WRAPPER

class CWRRTETransformer(nn.Module):
    """
    Main Wrapper
    - Оролтын урт sequence-г цонх болгож таслаад scan-аар дамжуулан гүйлгэнэ.
    """
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
        B, N = tokens.shape

        W = self.window_len
        O = self.overlap
        S = W - O

        n_win = math.ceil((N - W) / S) + 1 if N > W else 1

        # Оролтыг padding хийж цонхнуудад таслах
        pad_len = max(0, W + (n_win - 1) * S - N)
        pad_t   = jnp.pad(tokens, ((0, 0), (0, pad_len)), constant_values=pad_id)

        starts  = (jnp.arange(n_win) * S).astype(jnp.int32)
        wins    = jax.vmap(lambda s: jax.lax.dynamic_slice(pad_t, (0, s), (B, W)))(starts)

        # Global RoPE cache
        freqs = precompute_freqs_cis(self.embed_dim // self.num_heads, int(self.max_seq_len + 512))

        # Window cell-д өгөх тохиргоо
        cfg = dict(
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
            deterministic     = deterministic,
            l2_memory_size    = l2_memory_size
        )

        class ScanLoopWrapper(nn.Module):
            @nn.compact
            def __call__(self, carry, inputs):
                tokens_w, win_start = inputs
                cell = CWRRTEWindowCell(**cfg)
                return cell(carry, tokens_w, win_start, freqs)

        ScanCell = nn.scan(
            ScanLoopWrapper,
            variable_broadcast = "params",
            split_rngs         = {"params": False, "dropout": True},
            in_axes            = 0,
            out_axes           = 0
        )

        # Scan эхлэл төлөв (carry init)
        ik = jax.random.PRNGKey(99)
        k1, k2, k3, k4, k5 = jax.random.split(ik, 5)

        init = (
            jnp.zeros((B, O, self.embed_dim)),                                 # mem_emb
            jnp.zeros((B, O), dtype=jnp.int32),                                # mem_ids

            # summary-уудыг noise-тэй эхлүүлэх (NaN/zero-ийн эмзэг байдлаас сэргийлэх)
            jax.random.normal(k1, (B, self.embed_dim)) * 0.01,                 # ssum_fast
            jax.random.normal(k2, (B, self.embed_dim)) * 0.01,                 # ssum_slow

            # L1 slots
            jax.random.normal(k3, (B, epi_num_slots, self.embed_dim)) * 0.02,  # l1 keys
            jax.random.normal(k4, (B, epi_num_slots, self.embed_dim)) * 0.01,  # l1 vals
            jnp.ones((B, epi_num_slots)) * 1e3,                                # l1 age
            jnp.zeros((B, epi_num_slots)),                                     # l1 strength

            # L2 bank
            jnp.zeros((B, l2_memory_size, self.embed_dim)),                    # l2 bank
            jnp.zeros((B,), dtype=jnp.int32),                                  # l2 ptr

            # last_key + position base
            jax.random.normal(k5, (B, self.embed_dim)) * 0.01,                 # last_key
            jnp.array(0, dtype=jnp.int32)                                      # pos_base
        )

        _, (logits_per_win, aux_per_win) = ScanCell(name="scan_main")(init, (wins, starts))

        # Цонхнуудын logits-ийг нэг урт дараалал болгон нэгтгэх
        logits0 = logits_per_win[0]
        if n_win > 1:
            tail   = logits_per_win[1:, :, O:, :].transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            logits = jnp.concatenate([logits0, tail], axis=1)
        else:
            logits = logits0

        # aux-ийг цонхнууд дээр дундажлах
        aux = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), aux_per_win)

        return logits[:, :N, :], aux


# Машин сургалтын давталт

model = CWRRTETransformer(
    vocab_size        = vocab_size,
    embed_dim         = embed_dim,
    num_layers        = num_layers,
    num_heads         = num_heads,
    window_len        = cwr_window_len,
    overlap           = cwr_overlap,
    engram_vocab_size = engram_vocab_size,
    engram_ngram_n    = engram_ngram_n,
    engram_num_heads  = engram_num_heads,
    max_seq_len       = max_seq_len
)

@jax.jit
def train_step(state, batch, rng):
    # Машин сургалтын нэг алхам (forward + loss + backward)
    d_rng, n_rng = jax.random.split(rng)

    def loss_fn(p):
        logits, aux = model.apply(
            {"params": p},
            batch[:, :-1],
            deterministic=False,
            rngs={"dropout": d_rng}
        )

        ce   = optax.softmax_cross_entropy_with_integer_labels(logits[:, :batch.shape[1]-1], batch[:, 1:])
        mask = (batch[:, 1:] != pad_id).astype(jnp.float32)

        # Үндсэн text loss
        text_loss = jnp.sum(ce * mask) / (jnp.sum(mask) + 1e-6)

        # Санах ойн удирдлагын loss-ууд
        w_mean, r_mean, c_nll = aux

        w_reg = (w_mean - sgrm_target_rate) ** 2
        r_reg = (r_mean - recall_target_rate) ** 2

        total = text_loss \
            + (sgrm_budget_weight   * w_reg) \
            + (recall_budget_weight * r_reg) \
            + (contrastive_weight   * c_nll)

        return total

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss, n_rng

@jax.jit
def predict_step(params, inputs):
    # Inference (deterministic)
    return model.apply({"params": params}, inputs, deterministic=True)[0]

def generate(params, prompt, gen_len=100, temp=0.8):
    """
    Текст үүсгэх
    - JIT recompilation хийхээс сэргийлж тогтмол урттай (padded) оролт өгнө.
    """
    ids = encode_text(prompt)
    if ids and (ids[-1] == eos_id):
        ids.pop()

    print(f"Generating: '{prompt}'")

    MAX_INFER = sft_long_seq_len

    for _ in range(gen_len):
        curr = len(ids)
        if curr >= MAX_INFER:
            break

        inp = np.full((MAX_INFER,), pad_id, dtype=np.int32)

        # Хэрэв хэт урт бол сүүлийн MAX_INFER хэсгийг барина
        real_ctx = ids[-MAX_INFER:] if curr > MAX_INFER else ids
        L = len(real_ctx)
        inp[:L] = real_ctx

        logits = predict_step(params, jnp.array([inp]))

        # PAD/BOS-ийг сонгогдохоос хамгаалах
        next_l = logits[0, L-1, :].at[pad_id].set(-1e9).at[bos_id].set(-1e9)

        probs  = np.exp(np.array(next_l) / temp)
        probs  = probs / (np.sum(probs) + 1e-9)

        nxt = np.random.choice(len(probs), p=probs)
        ids.append(int(nxt))

        if nxt == eos_id:
            break

    return decode_ids(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=sft_total_steps)
    args = p.parse_args()

    print(f"\n{'='*60}\n  CWRRTESOUP V3\n{'='*60}\n")

    rng = jax.random.PRNGKey(seed)
    r1, r2 = jax.random.split(rng)

    dummy_input = jnp.zeros((1, sft_long_seq_len), dtype=jnp.int32)
    params      = model.init(r1, dummy_input)["params"]

    print(f"Params: {sum(x.size for x in jax.tree_util.tree_leaves(params))/1e6:.2f}M")

    # Optimizer
    sched = optax.warmup_cosine_decay_schedule(
        init_value   = 1e-6,
        peak_value   = sft_learning_rate,
        warmup_steps = sft_warmup_steps,
        decay_steps  = args.steps,
        end_value    = 1e-6
    )

    opt = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(sched, weight_decay=weight_decay)
    )

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    t0 = time.time()

    for s in range(1, args.steps + 1):
        idx   = np.random.randint(0, len(corpus_ids) - sft_long_seq_len - 1, sft_batch_size)
        batch = jnp.array(np.stack([corpus_ids[i:i+sft_long_seq_len+1] for i in idx]))

        state, loss, r2 = train_step(state, batch, r2)

        if s % sft_loss_freq == 0:
            print(f"Step {s:5d} | Loss: {loss:.4f} | Time: {time.time()-t0:.1f}s")

        if s % sft_sample_freq == 0:
            print("\n" + "-" * 40)
            st = generate(state.params, sample_prompt_text, sample_gen_len, sample_temp)
            print(f"Gen: {st}")
            speak_async(st, voice=tts_voice, speed=tts_speed, amp=tts_amp, enabled=tts_enabled)
            print("-" * 40 + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
