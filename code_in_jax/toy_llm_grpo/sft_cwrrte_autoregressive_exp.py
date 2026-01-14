#
#  CWRRTE: A Cross-Window Recurrent Transformer with Conditional Engram Memory
#
#  АРХИТЕКТУРЫН ТАЙЛБАР:
#   Энэ модель урт текстийг боловсруулахдаа Recurrent болон Retrieval (Хайлт) 
#   аргуудыг хослуулсан эрлийз архитектур юм. Үндсэн гурван бүрэлдэхүүн хэсэгтэй:
#
#   1. Recurrent Memory (mem):
#      Өмнөх цонхны сүүлийн хэсгийг (overlap) дараагийн цонх руу дамжуулж,
#      дарааллын тасралтгүй байдлыг хангана.
#
#   2. Global Summary (ssum):
#      Текстийн ерөнхий агуулгыг нэг вектор руу шахаж, урт хугацааны контекст
#      хадгалах үүрэгтэй.
#
#   3. Conditional Engram Memory (Engram санах ой):
#      DeepSeek-ийн арга буюу static knowledge хадгалах хэсэг.
#      Текст дэх N-gram (үгсийн хоршил)-ыг хэш болгон хувиргаж, томоохон
#      хүснэгтээс (Lookup Table) холбогдох мэдээллийг шууд татаж авна.
#      Энэ нь моделийг цээжлэх ачааллаас чөлөөлж, логик сэтгэлгээнд төвлөрүүлдэг.
#
#  Хэсгүүд
#   - Vectorized Rolling Hash : N-gram хэшийг Python давталтгүйгээр, GPU дээр зэрэгцүүлэн бодно.
#   - KV-Bank Integration     : Engram санах ойг Attention механизмд нэмэлт Key/Value болгон оруулна.
#   - RoPE                    : Байршлын векторыг дарааллын эхнээс зөв хуваарилна.
#   - JAX Scan                : Цонх хоорондын шилжилтийг өндөр хурдаар гүйцэтгэнэ.
#
#
#  Лавлагаа:
#   - DeepSeek, Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
#     https://www.arxiv.org/pdf/2601.07372
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

# Engram Санах ой
engram_vocab_size     = 100000 # Энграм хүснэгтийн хэмжээ (Slot count)
engram_ngram_n        = 4      # Хэдэн тэмдэгт харж хэш хийх вэ (N-gram)
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
    # Векторыг complex тоо болгон хувиргах
    x_complex = jax.lax.complex(x[..., 0::2], x[..., 1::2])
    # Эргүүлэх үйлдэл (Rotation)
    x_rotated = x_complex * freq_cis
    # Буцаагаад бодит тоо болгох
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1).reshape(B, T, H, D)
    return x_out

def precompute_freqs_cis(dim, max_len, theta=10000.0):
    """RoPE давтамжийн матрицыг урьдчилан тооцоолох"""
    freqs     = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t         = jnp.arange(max_len)
    freqs     = jnp.outer(t, freqs)     # (T, Dim/2)
    freqs_cis = jnp.exp(1j * freqs)     # e^(ix)
    return freqs_cis[None, :, None, :]


# VECTORIZED ENGRAM MODULE (Engram санах ой)

class VectorizedEngram(nn.Module):
    vocab_size   : int
    embed_dim    : int
    memory_size  : int   = 100000   # Санах ойн хүснэгтийн хэмжээ
    ngram_n      : int   = 4        # N-gram урт
    dropout_rate : float = 0.05     # Хэт дасгахаас сэргийлэх

    def setup(self):
        # Статик мэдлэг хадгалах том хэмжээний Embedding хүснэгт
        self.memory_table = self.param(
            "engram_table",
            nn.initializers.normal(stddev=0.02),
            (self.memory_size, self.embed_dim)
        )
        # Санах ойг хэр зэрэг ашиглахыг шийдэх суралцдаг хаалга (Gate)
        # Анхдагч утга бага (-2.0) байх нь сургалтын эхэнд саад болохгүй байхад тустай
        self.gate_logit = self.param("engram_gate", nn.initializers.constant(-2.0), (self.embed_dim,))

        # Анхны тоонууд үүсгэх (Deterministic Primes) - нэг удаа урьдчилан бэлдэнэ
        def get_primes(n):
            ps = []
            x  = 131
            for _ in range(n):
                ps.append(x)
                x = (x * 31) + 1
            return jnp.array(ps, dtype=jnp.uint32)

        self.primes = get_primes(self.ngram_n)

    @nn.compact
    def __call__(self, current_ids, prev_ids_overlap, deterministic=True):
        """
        current_ids      : (B, W) - Одоогийн цонхны токенууд
        prev_ids_overlap : (B, O) - Өмнөх цонхны төгсгөл (Context)
        """
        B, W = current_ids.shape
        O    = prev_ids_overlap.shape[1]

        # PAD токенууд хэшлэлтийг эвдэхээс сэргийлэх (Collision багасгана)
        current_ids      = jnp.where(current_ids      == pad_id, 0, current_ids)
        prev_ids_overlap = jnp.where(prev_ids_overlap == pad_id, 0, prev_ids_overlap)

        # Бүрэн контекст үүсгэх, Overlap + Current
        full_seq = jnp.concatenate([prev_ids_overlap, current_ids], axis=1).astype(jnp.uint32)

        # Vectorized Rolling Hash тооцоолол
        # Давталт бүрт N-gram-ийн нэг байрлалыг тооцоолж нийлбэрийг олно
        primes    = self.primes
        hash_sum  = jnp.zeros((B, W), dtype=jnp.uint32)
        start_idx = O    # Одоогийн цонх эхлэх индекс

        for i in range(self.ngram_n):
            # i дахь байрлалын шилжилтийг тооцох
            s_start = start_idx - i
            s_end   = full_seq.shape[1] - i

            # Холбогдох хэсгийг зүсэж авах
            chunk = full_seq[:, s_start:s_end]

            # Анхны тоогоор үржүүлж нэмэх (uint32 overflow аяндаа хийгдэнэ)
            hash_sum = hash_sum + (chunk * primes[i])

        # Хүснэгтээс хайх (Lookup)
        lookup_indices = hash_sum % self.memory_size
        retrieved_emb  = self.memory_table[lookup_indices] # (B, W, D)

        # Gating (Тохируулга)
        # Модель энэ санах ойг хэр чухал вэ гэдгийг өөрөө сурна
        gate = jax.nn.sigmoid(self.gate_logit)
        out  = retrieved_emb * gate

        # Dropout (Сургалтын үед санамсаргүйгээр заримыг нь унтраана)
        out = nn.Dropout(self.dropout_rate, deterministic=deterministic)(out)

        return out


# TRANSFORMER BLOCKS (ATTENTION & MLP)

class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, kv=None, freqs_cis=None, deterministic=True):
        # Хэрэв kv өгөгдөөгүй бол өөрөө өөртөө attend хийнэ
        if kv is None: kv = x

        B, Tq, C = x.shape
        _, Tk, _ = kv.shape     # KV нь (Mem + Engram + Current) учир урт байна
        head_dim = self.embed_dim // self.num_heads

        # Q, K, V прожекц
        q = nn.Dense(self.embed_dim, name="q_proj")(x)
        k = nn.Dense(self.embed_dim, name="k_proj")(kv)
        v = nn.Dense(self.embed_dim, name="v_proj")(kv)

        # Multi-head хэлбэрт оруулах
        q = q.reshape(B, Tq, self.num_heads, head_dim)
        k = k.reshape(B, Tk, self.num_heads, head_dim)
        v = v.reshape(B, Tk, self.num_heads, head_dim)

        # RoPE, Байршлын мэдээлэл нэмэх
        if freqs_cis is not None:
            # Дарааллын эхнээс эхэлж (0-ээс) зөв уртыг таслах хэрэгтэй
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
            # Mask ашиглан ирээдүйг болон зөвшөөрөгдөөгүй хэсгийг хаах
            attn_weights = jnp.where(mask, attn_weights, -1e9)

        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(0.1, deterministic=deterministic)(attn_probs)

        # Гаралт
        out = jnp.matmul(attn_probs, v)
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
        # Pre-LayerNorm бүтэц
        norm_x  = nn.LayerNorm()(x)
        # KV нь тусдаа ирж байгаа тул түүнийг бас нормалчлах шаардлагатай байж болно,
        # гэхдээ энд KV нь аль хэдийн бэлтгэгдсэн ирж байгаа гэж үзлээ.
        norm_kv = kv if kv is not None else norm_x 
        
        # Attention хэсэг
        attn_out = CausalSelfAttention(self.embed_dim, self.num_heads)(
            norm_x, mask=mask, kv=norm_kv, freqs_cis=freqs_cis, deterministic=deterministic
        )
        x = x + attn_out
        
        # MLP хэсэг
        x = x + MLP(self.embed_dim)(nn.LayerNorm()(x), deterministic=deterministic)
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
    deterministic     : bool = True

    @nn.compact
    def __call__(self, carry, tokens_w):
        # Carry задлах
        # mem_emb : Өмнөх цонхны overlap хэсгийн embedding (Recurrence-д зориулагдсан)
        # mem_ids : Өмнөх цонхны overlap хэсгийн ID-нууд (Engram хэш хийхэд зориулагдсан)
        # ssum    : Global Summary вектор
        mem_emb, mem_ids, ssum = carry 

        B, T = tokens_w.shape
        O    = self.overlap

        # Одоогийн оролтын Embedding
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens_w)
        
        # Engram Санах ойгоос мэдээлэл татах (Retrieval)
        # Engram нь KV-Bank-д нэмэгдэх тул түүнийг тусад нь боловсруулна
        engram_emb = VectorizedEngram(
            vocab_size  = self.vocab_size, 
            embed_dim   = self.embed_dim,
            memory_size = self.engram_vocab_size,
            ngram_n     = self.engram_ngram_n
        )(tokens_w, mem_ids, deterministic=self.deterministic) # Гаралт: (B, T, D)

        # Memory боловсруулах
        # Recurrent memory-г adapter-аар дамжуулж хэвийн болгоно
        mem_processed = nn.Dense(self.embed_dim, name="mem_adapter")(mem_emb)
        mem_processed = nn.LayerNorm(name="mem_norm")(mem_processed)

        # Summary Injection (Хураангуйг шингээх)
        # Суралцдаг Alpha gate ашиглан оролтод summary мэдээллийг нэмнэ
        alpha     = jax.nn.sigmoid(self.param("alpha_gate", nn.initializers.constant(_logit(self.alpha_init)), (self.embed_dim,)))
        ssum_proj = nn.Dense(self.embed_dim, use_bias=False)(ssum)
        x = x + (ssum_proj[:, None, :] * alpha[None, None, :])

        # KV-Bank байгуулах 
        # Attention механизмд харагдах дарааллыг бүтээнэ
        # Дараалал : [Memory (O) | Engram (T) | Current (T)]
        # Энэ дарааллаар залгах нь RoPE байршлыг зөв хадгалахад чухал
        kv_seq = jnp.concatenate([mem_processed, engram_emb, x], axis=1)

        # Mask бэлдэх
        # Бидний үүсгэсэн KV дараалал дээр хэн юуг харж болохыг тодорхойлно.
        # Query урт = T (зөвхөн одоогийн цонх)
        # Key урт   = O + T + T (Mem + Engram + Current)
        
        # Causal Mask (Гурвалжин), Одоогийн токен зөвхөн өмнөхөө харна
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        
        # Memory Mask, Санах ой бүхэлдээ харагдана 
        mem_mask    = jnp.ones((T, O), dtype=bool)
        
        # Engram Mask, Engram нь одоогийн токентой зэрэгцэж байгаа тул
        # тухайн агшин дахь токен өмнө татагдсан Engram мэдээллийг харж болно
        engram_mask = causal_mask 
        
        # Бүтэн mask нэгтгэх
        full_mask = jnp.concatenate([mem_mask, engram_mask, causal_mask], axis=1) # (T, O+T+T)

        # Padding Mask (Хоосон зайг харахгүй байх)
        valid_curr = (tokens_w != pad_id)
        valid_mem  = jnp.ones((B, O), dtype=bool)
        valid_eng  = valid_curr   # Engram нь токен байвал хүчинтэй
        
        valid_k = jnp.concatenate([valid_mem, valid_eng, valid_curr], axis=1)
        
        # Эцсийн Mask, (B, 1, T, Total_KV_Len)
        mask = full_mask[None, None, :, :] & valid_k[:, None, None, :]

        # RoPE давтамж бэлдэх
        # Нийт KV дарааллын уртад тааруулж бэлдэнэ
        total_kv_len = O + T + T
        freqs_cis = precompute_freqs_cis(self.embed_dim // self.num_heads, total_kv_len + 32)

        # Transformer Давхаргууд
        curr_x = x
        for i in range(self.num_layers):
            curr_x = Block(self.embed_dim, self.num_heads)(
                curr_x, mask=mask, kv=kv_seq, freqs_cis=freqs_cis, deterministic=self.deterministic
            )

        curr_x = nn.LayerNorm()(curr_x)

        # Дараагийн төлөвийг хадгалах (Update State)
        new_mem_emb = curr_x[:, -O:, :]
        new_mem_ids = tokens_w[:, -O:]    # ID-г дараагийн Engram хэшлэлд зориулж хадгална
        
        # Summary шинэчлэх логик
        # Одоогийн цонхны дунджийг олох
        out_mask = (tokens_w != pad_id).astype(jnp.float32)[:, :, None]
        win_sum  = jnp.sum(curr_x * out_mask, axis=1) / (jnp.sum(out_mask, axis=1) + 1e-6)
        
        # Шинэчлэх хувийг шийдэх Lambda gate
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

    @nn.compact
    def __call__(self, tokens_long, deterministic=True):
        # tokens_long, (B, Total_Seq_Len)
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
        windows = jax.vmap(lambda s: jax.lax.dynamic_slice(tokens_pad, (0, s), (B, W)))(starts)

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
        init_mem_ids = jnp.zeros((B, O), dtype=jnp.int32) # Эхний Engram хэшлэлд зориулсан ID
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
            deterministic     = deterministic
        )((init_mem_emb, init_mem_ids, init_ssum), windows)

        # Цонхнуудын гаралтыг буцааж нэг урт дараалал болгох
        out = logits_windows[0] # Эхний цонх бүтэн

        if n_win > 1:
            # Дараагийн цонхнуудаас зөвхөн шинэ хэсгийг нь авч залгана
            rest = logits_windows[1:, :, O:, :]
            rest = rest.transpose(1, 0, 2, 3).reshape(B, -1, self.vocab_size)
            out  = jnp.concatenate([out, rest], axis=1)

        return out[:, :N, :]


# СУРГАЛТ БОЛОН GENERATE ХИЙХ ФУНКЦУУД 

# Моделийн instance үүсгэх
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

@jax.jit
def train_step(state, batch, rng):
    """Нэг сургалтын алхам (Loss тооцох + Gradient шинэчлэх)"""
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(p):
        logits = model.apply(
            {"params": p},
            batch[:, :-1], # Оролт
            deterministic=False,
            rngs={"dropout": dropout_rng}
        )
        labels = batch[:, 1:] # Зорилтот утга
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

        # Padding хийж оролтыг тогтмол хэмжээтэй болгох (JIT-д зориулж)
        pad_len = max_len - curr_len
        inp_np  = np.array(token_ids + [pad_id] * pad_len, dtype=np.int32)
        inp_jax = jnp.array([inp_np])

        logits = predict_step_jit(params, inp_jax)
        next_token_logits = logits[0, curr_len - 1, :]
        
        # Тусгай токенуудыг сонгохгүй байхаар тохируулах
        next_token_logits = next_token_logits.at[pad_id].set(-1e9)
        next_token_logits = next_token_logits.at[bos_id].set(-1e9)
        
        # Sampling хийх
        probs = np.exp(np.array(next_token_logits) / temp)
        probs_sum = np.sum(probs)
        if probs_sum == 0 or np.isnan(probs_sum): probs = np.ones_like(probs) / len(probs)
        else: probs /= probs_sum
        
        next_id = np.random.choice(len(probs), p=probs)
        token_ids.append(next_id)
        
        if next_id == eos_id: break
            
    return decode_ids(token_ids)


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
    print("  CWRRTE: Cross-Window Recurrent Transformer + Engram")
    print(f"  Steps: {args.steps} | Batch: {args.batch} | SeqLen: {args.seq_len}")
    print(f"  Engram Size: {engram_vocab_size} | N-gram: {engram_ngram_n}")
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

    # Сургалтын процесс
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        # Санамсаргүй байдлаар өгөгдөл сонгох
        starts    = np.random.randint(0, len(corpus_ids) - args.seq_len - 1, args.batch)
        batch_np  = np.stack([corpus_ids[s : s + args.seq_len + 1] for s in starts])
        batch_jax = jnp.asarray(batch_np, dtype=jnp.int32)

        state, loss, rng = train_step(state, batch_jax, rng)

        if step % args.loss_freq == 0:
            dt = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s")
        
        if step % args.sample_freq == 0:
            print("\n" + "-"*40)
            sample_text = generate(state.params, sample_prompt_text, gen_len=sample_gen_len, temp=sample_temp)
            print(f"ГАРСАН ҮР ДҮН: {sample_text}")
            print("-"*40 + "\n")

    print("Сургалт амжилттай дууслаа!")

if __name__ == "__main__":
    main()