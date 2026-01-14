#
#  MSWIDT TRANSFORMER (SFT)
#
#  MSWIDT : Mosaic Shifted-Window Iterative Diffusion Transformer
#
#  ЗОРИЛГО:
#   Generative Masked Diffusion архитектурын туршилт.
#
#  MOSAIC MECHANISM:
#   - Mosaic Tiles (Local Mixing)
#   - Weaving (Global Mixing via Shift)
#   - Iterative Refinement (Shared Weights)
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


# CONFIGURATION

# Data
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# Architecture
msw_window_len        = 64
msw_num_phases        = 6      
msw_embed_dim         = 256
msw_num_heads         = 8

# Training
diff_total_steps      = 10000
diff_seq_len          = 512
diff_batch_size       = 16
diff_learning_rate    = 3e-4
diff_warmup_steps     = 200

# Sampling
log_loss_freq         = 50
log_sample_freq       = 500
sample_gen_steps      = 40     
sample_prompt_text    = "Once upon a time"
sample_temp           = 1.0

# Optimization
max_grad_norm         = 1.0
weight_decay          = 0.01

np.random.seed(seed)
random.seed(seed)


# DATA PREPARATION

print(">>> Өгөгдлийг уншиж байна...")
if not os.path.exists(dataset_path):
    print("Анхаар: Dataset олдсонгүй. Хиймэл текст ашиглана.")
    dummy_vocab = ["robot", "girl", "boy", "run", "fly", "happy", "sad", "sky", "blue", "red"]
    raw_text    = ""
    for _ in range(5000):
        sent = " ".join(random.choices(dummy_vocab, k=12)) + ". "
        raw_text += sent + end_of_text_token
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

# Tokenizer
all_stories  = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 10]
unique_chars = sorted(list(set("".join(all_stories))))

PAD, MASK, EOS = "<PAD>", "<MASK>", "<EOS>"
chars          = [PAD, MASK, EOS] + unique_chars
char_to_id     = {c: i for i, c in enumerate(chars)}
id_to_char     = {i: c for c, i in char_to_id.items()}
vocab_size     = len(chars)

pad_id, mask_id, eos_id = char_to_id[PAD], char_to_id[MASK], char_to_id[EOS]

def encode_text(text):
    return [char_to_id.get(ch, pad_id) for ch in text] + [eos_id]

def decode_ids(ids):
    return "".join([id_to_char[int(i)] for i in ids if int(i) not in [pad_id, eos_id, mask_id]])

def decode_visual(ids):
    res = []
    for i in ids:
        if int(i) == mask_id: res.append("_")
        elif int(i) in [pad_id, eos_id]: continue
        else: res.append(id_to_char[int(i)])
    return "".join(res)

flat_tokens = []
print(">>> Токенчилж байна...")
for s in all_stories:
    flat_tokens.extend(encode_text(s))
corpus_ids = np.array(flat_tokens, dtype=np.int32)
print(f"Stats: Vocab={vocab_size}, Tokens={len(corpus_ids)}")


# MSWIDT BLOCK & MODEL

class MSWIDTBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, D  = x.shape
        residual = x
        x        = nn.LayerNorm()(x)

        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(x)
        v = nn.Dense(self.embed_dim)(x)

        head_dim = self.embed_dim // self.num_heads
        q = q.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        attn_probs   = jax.nn.softmax(attn_weights, axis=-1)
        out = jnp.matmul(attn_probs, v).transpose(0, 2, 1, 3).reshape(B, T, D)
        
        x = nn.Dense(self.embed_dim)(out)
        x = nn.Dropout(0.1, deterministic=deterministic)(x)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embed_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.Dropout(0.1, deterministic=deterministic)(x)
        x = x + residual
        
        return x

class MSWIDT(nn.Module):
    vocab_size  : int
    embed_dim   : int
    num_heads   : int
    window_len  : int
    num_phases  : int

    @nn.compact
    def __call__(self, x, mask_ratio_t, deterministic=True):
        B, N = x.shape
        W    = self.window_len
        
        h       = nn.Embed(self.vocab_size, self.embed_dim)(x)
        pos_emb = nn.Embed(8192, self.embed_dim)(jnp.arange(N)[None, :])
        h       = h + pos_emb
        
        # Mask Indicator 
        is_masked      = (x == mask_id).astype(jnp.int32)
        mask_indicator = nn.Embed(2, self.embed_dim)(is_masked)
        h = h + mask_indicator
        
        # Time Embedding
        t_proj = nn.Dense(self.embed_dim)(mask_ratio_t)
        t_proj = nn.gelu(t_proj)
        h      = h + t_proj[:, None, :]

        block      = MSWIDTBlock(self.embed_dim, self.num_heads)
        phase_embs = self.param("phase_embs", nn.initializers.normal(stddev=0.02), (self.num_phases, self.embed_dim))

        for phase in range(self.num_phases):
            do_shift  = (phase % 2 == 1)
            shift_amt = W // 2
            
            h = h + phase_embs[phase][None, None, :]
            
            phase_input = h
            if do_shift: h = jnp.roll(h, shift=-shift_amt, axis=1)
            
            num_windows = N // W
            h_windows   = h.reshape(B * num_windows, W, self.embed_dim)
            h_windows   = block(h_windows, deterministic=deterministic)
            h           = h_windows.reshape(B, N, self.embed_dim)
            
            if do_shift: h = jnp.roll(h, shift=shift_amt, axis=1)
            
            # Scaled Residual
            h = (phase_input + h) * 0.5 

        logits = nn.Dense(self.vocab_size)(h)
        return logits


# TRAINING (BERT STYLE 80/10/10)

model = MSWIDT(
    vocab_size = vocab_size,
    embed_dim  = msw_embed_dim ,
    num_heads  = msw_num_heads ,
    window_len = msw_window_len,
    num_phases = msw_num_phases
)

@jax.jit
def train_step(state, batch, rng, step_count):
    # Санамсаргүй түлхүүрүүд
    rng, t_rng, mask_rng, prefix_rng, bert_rng, dropout_rng = jax.random.split(rng, 6)
    B, N = batch.shape
    
    # Mask Ratio (t)
    # Character level тул хэт хэцүү болгохгүй (max 50%)
    t = jax.random.uniform(t_rng, (B, 1), minval=0.10, maxval=0.50)
    
    # Uniform Random Masking (No Spans)
    # Span masking нь character-level дээр хэт хэцүү тул Uniform ашиглана.
    probs        = jax.random.uniform(mask_rng, (B, N))
    mask_indices = (probs < t)
    
    # Protection
    prefix_lens   = jax.random.randint(prefix_rng, (B, 1), 0, 40)
    token_indices = jnp.arange(N)[None, :]
    
    fillable = (token_indices >= prefix_lens) & (batch != eos_id) & (batch != pad_id)
    mask_indices = mask_indices & fillable
    
    # BERT Corruption Strategy (80/10/10)
    # Зөвхөн [MASK] тавих биш, загварыг "хуурах" арга.
    # 80% -> [MASK]
    # 10% -> Random Token
    # 10% -> Original Token (loss бодогдоно)
    
    bert_probs = jax.random.uniform(bert_rng, (B, N))
    
    # MASK-аар солих (0.0-с 0.8)
    replace_mask   = (bert_probs < 0.8) & mask_indices
    # Random-аар солих (0.8-с 0.9)
    random_token   = jax.random.randint(bert_rng, (B, N), 0, vocab_size)
    replace_random = (bert_probs >= 0.8) & (bert_probs < 0.9) & mask_indices
    
    # Input
    masked_input = jnp.where(replace_mask, mask_id, batch)
    masked_input = jnp.where(replace_random, random_token, masked_input)
    # Үлдсэн 10% нь mask_indices=True боловч batch дээрх утгаараа үлдэнэ 
    
    def loss_fn(p):
        logits = model.apply(
            {"params": p}, 
            masked_input, 
            t, 
            deterministic=False, 
            rngs={'dropout': dropout_rng}
        )
        
        loss_t     = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
        valid_mask = mask_indices.astype(jnp.float32)
        
        denom = jnp.maximum(jnp.sum(valid_mask), 1.0)
        loss  = jnp.sum(loss_t * valid_mask) / denom
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, rng


# GENERATION (CONFIDENCE-BASED ITERATIVE)

def generate(params, prompt, rng, steps=40):
    """
    Generation:
    алдаагаа засах боломжтой уян хатан iterative арга.
    """
    p_ids = encode_text(prompt)
    if p_ids[-1] == eos_id: p_ids.pop()
    
    current_ids = np.full(diff_seq_len, mask_id, dtype=np.int32)
    current_ids[:len(p_ids)] = p_ids
    
    print(f"MSWIDT Generating: '{prompt}'")
    
    for step in range(steps):
        progress     = step / steps
        ratio        = 1.0 - progress
        current_temp = 1.0 * (1.0 - progress) + 0.1 * progress # Бага зэрэг noise үлдээнэ
        
        rng, step_rng = jax.random.split(rng)
        inp_jax       = jnp.array([current_ids])
        
        t_signal = jnp.array([[ratio]]) 
        logits   = model.apply({"params": params}, inp_jax, t_signal, deterministic=True)
        
        # Sampling
        logits      = logits / current_temp
        sampled_ids = jax.random.categorical(step_rng, logits, axis=-1)[0]
        sampled_ids = np.array(sampled_ids)
        
        # Бүх байрлал дээр шинэчлэлт хийнэ (Monotonic биш)
        # Ингэснээр модель өмнөх алхамын алдаагаа (typo) засах боломжтой болно.
        # Гэхдээ Prompt хэсгийг хөндөхгүй.
        current_ids[len(p_ids):] = sampled_ids[len(p_ids):]
        
        # Confidence Re-masking
        probs       = jax.nn.softmax(logits, axis=-1)
        confidences = jnp.max(probs, axis=-1)[0]
        
        # Prompt/EOS/PAD-ийг өндөр confidence-тэй гэж үзнэ
        is_stable   = (current_ids == eos_id) | (current_ids == pad_id)
        confidences = jnp.where(is_stable, confidences + 100.0, confidences)
        confidences = confidences.at[:len(p_ids)].set(1e9)
        
        # Зорилтот mask-ийн тоо
        num_to_mask = int(ratio * (diff_seq_len - len(p_ids)))
        if num_to_mask <= 0:
            break
            
        # Хамгийн бага confidence-тэй хэсгүүдийг буцааж MASK болгоно
        mask_indices = np.argsort(np.array(confidences))[:num_to_mask]
        current_ids[mask_indices] = mask_id
        
        # Prompt restore
        current_ids[:len(p_ids)] = p_ids
        
        if step % 5 == 0:
            vis = decode_visual(current_ids[:60])
            print(f"Step {step+1:02d} (T={current_temp:.2f}) | {vis}...")

    return decode_ids(current_ids)


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=diff_total_steps)
    parser.add_argument("--batch", type=int, default=diff_batch_size)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  MSWIDT v8 - RESTORATION")
    print(f"  Strategy: BERT-style Corruption & Flexible Refinement")
    print(f"  Steps: {args.steps} | Batch: {args.batch} | SeqLen: {diff_seq_len}")
    print("="*60 + "\n")

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    dummy_in  = jnp.zeros((1, diff_seq_len), dtype=jnp.int32)
    dummy_t   = jnp.ones((1, 1), dtype=jnp.float32)
    variables = model.init(init_rng, dummy_in, dummy_t, deterministic=True)
    params    = variables["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model Params: {param_count/1e6:.2f}M\n")

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value   = 1e-7,
                peak_value   = diff_learning_rate,
                warmup_steps = diff_warmup_steps,
                decay_steps  = args.steps,
                end_value    = 1e-7
            ),
            weight_decay = weight_decay
        )
    )
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        idx       = np.random.randint(0, len(corpus_ids) - diff_seq_len, args.batch)
        batch     = np.stack([corpus_ids[i:i+diff_seq_len] for i in idx])
        batch_jax = jnp.asarray(batch, dtype=jnp.int32)

        state, loss, rng = train_step(state, batch_jax, rng, step)

        if step % log_loss_freq == 0:
            dt = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss:.4f} | Time: {dt:.1f}s")
        
        if step % log_sample_freq == 0:
            print("\n" + "-"*40)
            rng, gen_rng = jax.random.split(rng)
            gen_text = generate(state.params, sample_prompt_text, gen_rng, steps=sample_gen_steps)
            print(f"MSWIDT RESULT:\n{gen_text}")
            print("-"*40 + "\n")

    print("Сургалт амжилттай дууслаа!")

if __name__ == "__main__":
    main()