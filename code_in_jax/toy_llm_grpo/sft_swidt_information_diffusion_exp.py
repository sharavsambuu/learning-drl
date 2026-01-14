#
#  SWIDT TRANSFORMER (SFT)
#
#  SWIDT : Shifted Window Iterative Diffusion Transformer
#
#  ЗОРИЛГО:
#   - Урт текстийг боловсруулахдаа Iterative (давталтат) аргаар сайжруулах.
#   - Information Diffusion буюу мэдээлэл түгээх процессыг Window Shift (цонх шилжүүлэх) аргаар хийх.
#   - SFT (Supervised Fine-Tuning) буюу дараагийн үг таах даалгавар.
#
#  АРХИТЕКТУРЫН ОНЦЛОГ (SWIDT):
#   1. Iterative Refinement: Модель олон давхаргатай биш, харин нэг хүчирхэг давхаргыг
#      олон дахин (num_phases) ашиглаж үр дүнгээ сайжруулна. (Shared Weights)
#   2. Shifted Windows: 
#      - Phase 0, 2, 4... : Текстийг энгийнээр цонхнуудад хуваана [0-128, 128-256...]
#      - Phase 1, 3, 5... : Текстийг шилжүүлж хуваана [64-192, 192-320...]
#      Үүгээр цонхнуудын хоорондох хана хэрмийг нурааж мэдээллийг бүх текст даяар тараана.
#
#
#

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

import math
import random
import argparse
import time
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state


# ТОХИРГОО (HYPERPARAMETERS)

# Өгөгдөл
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# SWIDT Архитектур
swidt_window_len      = 128    # Нэг цонхны урт
swidt_num_phases      = 6      # Хэдэн удаа давтаж сайжруулах вэ? (Diffusion steps)
embed_dim             = 256    # Векторын хэмжээ
num_heads             = 8      # Attention толгойн тоо

# Сургалт (SFT)
sft_total_steps       = 5000
sft_seq_len           = 1024   # Сургах текстийн урт 
sft_batch_size        = 8
sft_learning_rate     = 3e-4
sft_warmup_steps      = 200

# Лог болон Туршилт
sft_loss_freq         = 20     # Loss хэвлэх давтамж
sft_sample_freq       = 200    # Текст үүсгэж шалгах давтамж
sample_gen_len        = 200    # Үүсгэх текстийн урт
sample_temp           = 0.8    # Sampling temperature

# Тогтворжилт
max_grad_norm         = 1.0
weight_decay          = 0.01

# Random seed
np.random.seed(seed)
random.seed(seed)


# ӨГӨГДӨЛ БЭЛТГЭХ 

print(">>> Өгөгдлийг уншиж байна...")
if not os.path.exists(dataset_path):
    print("Анхаар: Dataset олдсонгүй. Хиймэл туршилтын текст үүсгэж байна.")
    # Жишээ текст үүсгэх
    dummy_vocab = ["robot", "girl", "boy", "dog", "cat", "run", "jump", "happy", "sad", "big", "little"]
    raw_text    = ""
    for _ in range(3000):
        sent = " ".join(random.choices(dummy_vocab, k=15)) + ". "
        raw_text += sent + end_of_text_token
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

# Тэмдэгт түвшний tokenizer
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

# Бүх текстийг тоо болгох
flat_tokens = []
print(">>> Токенчилж байна...")
for s in all_stories:
    flat_tokens.extend(encode_text(s))
corpus_ids = np.array(flat_tokens, dtype=np.int32)

print(f"Dataset Stats: Vocab={vocab_size}, Total Tokens={len(corpus_ids)}")


# SWIDT МОДЕЛИЙН БҮРЭЛДЭХҮҮН ХЭСГҮҮД

# Causal Mask (ирээдүйг харахгүй байх)
def make_causal_mask(T):
    # (1, 1, T, T) хэлбэртэй mask
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    return mask[None, None, :, :]

class SWIDTBlock(nn.Module):
    """
    Transformer Block, энэ блокийг олон удаа (phases) давтаж ашиглана.
    """
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # x shape: (B_windows, Window_Len, Dim)
        
        # Attention
        residual = x
        x = nn.LayerNorm()(x)
        
        B, T, D = x.shape
        head_dim = self.embed_dim // self.num_heads

        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(x)
        v = nn.Dense(self.embed_dim)(x)

        # Multi-head split
        q = q.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention Score
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
            
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(0.1, deterministic=deterministic)(attn_probs)

        # Merge heads
        out = jnp.matmul(attn_probs, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        
        # Output projection
        x = nn.Dense(self.embed_dim)(out)
        x = nn.Dropout(0.1, deterministic=deterministic)(x)
        x = x + residual # Residual 1

        # Feed Forward (MLP)
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embed_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.Dropout(0.1, deterministic=deterministic)(x)
        x = x + residual # Residual 2

        return x

class SWIDTTransformer(nn.Module):
    """
    Shifted-Window Iterative Diffusion Transformer (SWIDT)
    """
    vocab_size  : int
    embed_dim   : int
    num_heads   : int
    window_len  : int
    num_phases  : int # Diffusion алхамын тоо

    @nn.compact
    def __call__(self, x, deterministic=True):
        # x: (Batch, SeqLen)
        B, N = x.shape
        W    = self.window_len
        
        # Embeddings
        # Token embedding
        h = nn.Embed(self.vocab_size, self.embed_dim)(x)
        
        # Positional embedding (Global)
        # Урт текст дээр байршлаа алдахгүйн тулд global position хэрэглэнэ
        pos_ids = jnp.arange(N)[None, :]
        pos_emb = nn.Embed(8192, self.embed_dim)(pos_ids) # Хангалттай том max_len
        h       = h + pos_emb

        # Shared Block (Үүнийг бүх phase дээр давтаж ашиглана)
        # Universal Transformer-тай төстэй санаа боловч Shifted Window нэмэгдсэн
        shared_block = SWIDTBlock(self.embed_dim, self.num_heads)
        
        # Causal Mask (Window доторх)
        # Цонх болгон дотроо ирээдүйгээ харахгүй байх ёстой
        local_mask = make_causal_mask(W)

        # Iterative Diffusion Loop
        # Phase бүрт мэдээлэл сайжирч (refine), цонхнуудын хооронд тархана (diffuse)
        
        # Энэ loop нь JAX-ийн unroll хийгддэг тул compile хийхэд асуудалгүй.
        for phase in range(self.num_phases):
            # Тэгш phase    (0, 2..): Энгийн цонх
            # Сондгой phase (1, 3..): Шилжсэн (Shifted) цонх
            do_shift  = (phase % 2 == 1)
            shift_amt = W // 2
            
            # Diffusion phase-д зориулсан residual connection 
            # Phase эхлэхийн өмнөх төлөвийг хадгална (Gradient flow-д хэрэгтэй)
            phase_input = h
            
            # Shift хийх (Rolling)
            if do_shift:
                # Зүүн тийш шилжүүлнэ. [A, B, C, D] -> [B, C, D, A]
                # Ингэснээр цонхны хил зааг өөрчлөгдөж, өмнө нь зах байсан токенууд голд орно.
                h = jnp.roll(h, shift=-shift_amt, axis=1)

            # Window-д хуваах
            # (Batch, SeqLen, Dim) -> (Batch * NumWindows, WindowLen, Dim)
            # SeqLen нь WindowLen-д хуваагддаг байх хэрэгтэй
            # Хэрэв хуваагдахгүй бол padding хийх шаардлагатай
            num_windows = N // W
            h_windows   = h.reshape(B * num_windows, W, self.embed_dim)

            # Block ажиллуулах (Parallel Processing)
            # Бүх цонх зэрэгцээ боловсруулагдана.
            h_windows = shared_block(h_windows, mask=local_mask, deterministic=deterministic)

            # Буцааж дэлгэх
            h = h_windows.reshape(B, N, self.embed_dim)

            # Un-shift (Shift-ийг буцаах)
            if do_shift:
                # Байранд нь буцааж оруулах. [B, C, D, A] -> [A, B, C, D]
                h = jnp.roll(h, shift=shift_amt, axis=1)
            
            # Iterative Residual
            # Өмнөх phase-ийн мэдээлэл дээр шинэ боловсруулалтыг нэмнэ.
            # Энэ бол diffusion буюу мэдээллийг аажмаар баяжуулах процесс юм.
            h = phase_input + h

        # Final Norm & Head
        h      = nn.LayerNorm()(h)
        logits = nn.Dense(self.vocab_size)(h)
        
        return logits


# СУРГАЛТ БОЛОН ТЕКСТ ҮҮСГЭХ ФУНКЦУУД

# Моделийг үүсгэх
model = SWIDTTransformer(
    vocab_size  = vocab_size,
    embed_dim   = embed_dim,
    num_heads   = num_heads,
    window_len  = swidt_window_len,
    num_phases  = swidt_num_phases
)

@jax.jit
def train_step(state, batch, rng):
    """SFT сургалтын нэг алхам"""
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(p):
        # Forward pass
        # batch[:, :-1] -> Оролт
        logits = model.apply(
            {"params": p},
            batch[:, :-1], 
            deterministic=False,
            rngs={"dropout": dropout_rng}
        )

        # batch[:, 1:] -> Labels (Дараагийн токен)
        labels = batch[:, 1:]
        
        # Loss тооцох (Cross Entropy)
        loss_t = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        
        # Padding токенуудыг loss-д тооцохгүй
        mask = (labels != pad_id).astype(jnp.float32)
        loss = jnp.sum(loss_t * mask) / (jnp.sum(mask) + 1e-6)
        
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_rng

def generate(params, prompt, gen_len=100, temp=0.8):
    """
    SWIDT загвараар текст үүсгэх
    SWIDT нь iterative модель тул токен бүрийг үүсгэхдээ бүх phase-ийг дахин ажиллуулна 
    """
    token_ids = list(encode_text(prompt))
    if token_ids[-1] == eos_id: token_ids.pop() # EOS байвал хасах

    # JIT compile-ийг дахин дахин хийхгүйн тулд тогтмол урттайгаар ажиллуулна
    # generate хийх үеийн хамгийн дээд урт
    MAX_GEN_CTX = sft_seq_len 
    
    print(f"Generating: '{prompt}'...")

    for _ in range(gen_len):
        curr_len = len(token_ids)
        if curr_len >= MAX_GEN_CTX: break

        # Оролтыг бэлдэх (Padding хийх)
        # SWIDT нь window-д хуваагддаг тул урт нь window_len-д хуваагддаг байвал сайн.
        # sft_seq_len нь 1024 (128*8) тул асуудалгүй.
        pad_len = MAX_GEN_CTX - curr_len
        inp_np  = np.array(token_ids + [pad_id] * pad_len, dtype=np.int32)
        inp_jax = jnp.array([inp_np]) # (1, SeqLen)

        # Моделийг ажиллуулах (Inference mode)
        logits = model.apply({"params": params}, inp_jax, deterministic=True)
        
        # Сүүлийн valid токений logit-ийг авах
        next_token_logits = logits[0, curr_len - 1, :]
        
        # Sampling logic, Pad болон BOS-ийг дарах
        next_token_logits = next_token_logits.at[pad_id].set(-1e9)
        next_token_logits = next_token_logits.at[bos_id].set(-1e9)
        
        probs = jax.nn.softmax(next_token_logits / temp)
        probs = np.array(probs)
        
        # Сонгох
        next_id = np.random.choice(len(probs), p=probs)
        token_ids.append(next_id)
        
        if next_id == eos_id:
            break
            
    return decode_ids(token_ids)


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=sft_total_steps)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SWIDT TRANSFORMER - TRAINING START")
    print(f"  Architecture: Shifted-Window Iterative Diffusion")
    print(f"  Window Len  : {swidt_window_len}")
    print(f"  Phases      : {swidt_num_phases} (Diffusion Steps)")
    print(f"  Seq Len     : {sft_seq_len}")
    print(f"  Batch Size  : {sft_batch_size}")
    print("="*60 + "\n")

    # Random Key
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Параметрүүд үүсгэх
    dummy_in  = jnp.zeros((1, sft_seq_len), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_in, deterministic=True)
    params    = variables["params"]

    # Параметрийн тоо
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Моделийн нийт параметр: {param_count/1e6:.2f}M")
    print(f"Shared Weights ашигласан тул параметр бага ч, тооцоолол нь {swidt_num_phases} дахин их")

    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value   = 1e-7,
                peak_value   = sft_learning_rate,
                warmup_steps = sft_warmup_steps,
                decay_steps  = args.steps,
                end_value    = 1e-7
            ),
            weight_decay=weight_decay
        )
    )

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Training Loop
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        # Batch бэлдэх
        # Текстийн санамсаргүй хэсгээс таслан авах
        starts    = np.random.randint(0, len(corpus_ids) - sft_seq_len - 1, sft_batch_size)
        batch_np  = np.stack([corpus_ids[s : s + sft_seq_len + 1] for s in starts])
        batch_jax = jnp.asarray(batch_np, dtype=jnp.int32)

        # Train Step
        state, loss, rng = train_step(state, batch_jax, rng)

        # Log
        if step % sft_loss_freq == 0:
            dt = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s")

        # Sample Generation (Үр дүнгээ харах)
        if step % sft_sample_freq == 0:
            print("\n" + "-"*40)
            sample_prompt = "Once upon a time"
            gen_text = generate(state.params, sample_prompt, gen_len=sample_gen_len, temp=sample_temp)
            print(f"SWIDT GENERATION (Step {step}):\n{gen_text}")
            print("-"*40 + "\n")

    print("Сургалт амжилттай дууслаа!")

if __name__ == "__main__":
    main()