#
# HIERARCHICAL LATENT STEERING WITH GRPO (Internal HRL)
#
# https://huggingface.co/papers/2512.20605
# https://arxiv.org/pdf/2512.20605
#
# Зорилго
#
# Энэ код нь LLM-ийн зан авирыг үг, токен бүрээр нь бус харин утга санаа буюу бодол-ын 
# түвшинд удирдах Hierarchical Reinforcement Learning (HRL) аргачлалыг хэрэгжүүлэн турших явдал юм.
#
#
# Энэ хэрэгжүүлэлт нь Google DeepMind болон AI Alignment судалгааны дараах
# ажлуудаас санаа авсан Hybrid архитектур юм
#
# 1. Director Architecture (Danijar Hafner et al., DeepMind):
#    - Том моделийг (Worker) жижиг модель (Manager) удирддаг бүтэц.
#    - Manager нь токен биш, харин Latent Goal вектор ялгаруулдаг.
#
# 2. Activation Engineering / Steering Vectors (Anthropic, DeepMind):
#    - Моделийн жинг өөрчлөхгүйгээр, NN-ий урсгал дунд (Residual Stream) 
#      вектор нэмэх замаар гаралтыг удирдах арга.
#
# 3. Temporal Abstraction (Sutton et al.):
#    - Шийдвэр гаргалтыг цаг хугацааны хувьд багцлах. 
#      Өөрөөр хэлбэл Controller нь алхам бүрд биш
#      харин K алхам тутамд нэг удаа шийдвэр гаргана.
#
#
# Хэрэгжүүлэлт
#
# A. SPLIT ARCHITECTURE (Хуваагдмал бүтэц):
#    - Base Model (Frozen): Хэлний зүй тогтлыг мэддэг бие буюу body нь.
#      Энэ моделийн жин сургалтын явцад өөрчлөгдөхгүй.
#    - Meta-Controller (Trainable): Утга санааг удирддаг тархи буюу mind нь.
#      Зөвхөн энэ жижиг сүлжээ л сургагдана.
#
# B. INTERNAL INTERVENTION (Дотоод оролцоо):
#    - Controller нь текст үүсгэдэггүй. Харин Base Model-ийн дунд давхарга
#      руу (Intervention Layer) Latent Z вектор шахдаг.
#    - Forward Pass:  h_modified = h_original + Projection(z)
#
# C. TEMPORAL PERSISTENCE (Цаг хугацааны тогтворжилт):
#    - Controller-ийн гаргасан Z вектор нь temporal_k алхмын турш хүчинтэй байна. 
#      Энэ нь моделийг нэг сэдэв/сэтгэл хөдлөлөө тууштай хадгалахад тусална.
#
# D. GRPO (Group Relative Policy Optimization):
#    - Controller-ийг сургахдаа орчин үеийн GRPO ашиглагдсан.
#    - Critic модель ашиглахгүйгээр, өөрийн үүсгэсэн олон хувилбар (Rollouts) дундаас 
#      сайныг нь шагнаж, мууг нь шийтгэх замаар сурдаг.
#      Токен дээр биш, Latent Z вектор дээр градиент тооцно.
#
#


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

import re
import gc
import math
import random
import jax
import optax
import flax
import numpy         as np
import flax.linen    as nn
from   jax           import numpy as jnp
from   flax.training import train_state


# CONFIGURATION

debug                 = True
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42

# SFT (Supervised Fine-Tuning)
sft_total_steps       = 8000    # Суурь мэдлэг олгох алхам
sft_batch_size        = 64
sft_seq_len           = 128
sft_learning_rate     = 5e-4
sft_warmup_steps      = 100
sft_sample_freq       = 500

# INTERNAL RL & GRPO Settings
grpo_total_updates    = 5000
group_size            = 8       # Бүлгийн хэмжээ (Жижиг GPU-д таарсан)
prompts_per_update    = 4
gen_len               = 128     # Үүсгэх текстийн урт
grpo_temp             = 0.9     # Текст үүсгэх температур
grpo_sample_freq      = 50

# Internal Control Parameters
latent_dim            = 32      # Удирдлагын векторын хэмжээ
temporal_k            = 4       # Нэг бодлыг хэдэн алхам барих вэ (Persistence)
intervention_layer    = 1       # Аль давхаргад нөлөөлөх вэ

# PPO Settings
ppo_epochs            = 2
mini_batch_size       = 8       # Gradient Accumulation-д зориулав
accum_steps           = 4       # Total batch = 4 * 8 = 32
clip_epsilon          = 0.2
grpo_lr               = 1e-4    # Controller сургах хурд
max_grad_norm         = 1.0
kl_beta               = 0.05    # KL шийтгэл
target_kl             = 0.05

# Architecture
prompt_len            = 16
model_max_len         = 256
num_layers            = 4
num_heads             = 4
embed_dim             = 128
dropout_rate          = 0.1

# Vocabularies
happy_vocab = ["happy", "joy", "joyful", "smile", "smiled", "laugh", "laughed", "love", "loved", "kind", "nice", "fun", "good", "great", "amazing", "wonderful", "excited", "brave", "bright", "safe", "friend", "friends"]
sad_vocab   = ["sad", "cry", "cried", "bad", "angry", "mad", "hurt", "scary", "afraid", "fear", "dark", "hate", "hated", "mean", "alone", "lost", "dead", "death", "kill", "killed"]
negations   = ["not", "no", "never", "don't", "can't", "won't", "neither", "nor"]

np.random.seed(seed)
random.seed(seed)


# DATASET AND TOKENIZATION

if not os.path.exists(dataset_path):
    print(f"Анхаар {dataset_path} файл олдсонгүй Жишээ текст ашиглана")
    raw_text = "Once upon a time there was a happy robot. It loved to smile. " * 2000
else:
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

all_stories    = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 0]
unique_chars   = sorted(list(set("".join(all_stories))))
PAD, BOS       = "<PAD>", "<BOS>"
chars          = [PAD, BOS] + unique_chars
char_to_id     = {c: i for i, c in enumerate(chars)}
id_to_char     = {i: c for c, i in char_to_id.items()}
vocab_size     = len(chars)
pad_id, bos_id = char_to_id[PAD], char_to_id[BOS]

def encode_text(text):
    return [bos_id] + [char_to_id.get(ch, pad_id) for ch in text]

def decode_ids(ids):
    return "".join([id_to_char[int(i)] for i in ids if int(i) not in [pad_id, bos_id]])

corpus_ids = np.array(encode_text("\n".join(all_stories)), dtype=np.int32)

print(f"Vocab Size  : {vocab_size}")
print(f"Total Chars : {len(corpus_ids)}")


# MODEL ARCHITECTURE (With Meta-Controller)

class MetaController(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # Hidden state -> Latent Distribution (Gaussian)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        mu      = nn.Dense(self.latent_dim)(x)
        log_std = nn.Dense(self.latent_dim)(x)
        # Numerical stability
        log_std = jnp.clip(log_std, -5, 2)
        return mu, log_std

class CausalSelfAttention(nn.Module):
    embed_dim: int; num_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        B, T, C  = x.shape
        head_dim = self.embed_dim // self.num_heads
        
        q = nn.Dense(self.embed_dim)(x).reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = nn.Dense(self.embed_dim)(x).reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = nn.Dense(self.embed_dim)(x).reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        if mask is not None:
            attn = jnp.where(mask, attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return nn.Dense(self.embed_dim)(out)

class MLP(nn.Module):
    embed_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        return x

class Block(nn.Module):
    embed_dim: int; num_heads: int
    @nn.compact
    def __call__(self, x, mask=None):
        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(nn.LayerNorm()(x), mask)
        x = x + MLP(self.embed_dim)(nn.LayerNorm()(x))
        return x

class HierarchicalTransformer(nn.Module):
    vocab_size: int; embed_dim: int; num_layers: int; num_heads: int; max_len: int

    @nn.compact
    def __call__(self, tokens, intervention_latents=None, stop_at_layer=None):
        b, t = tokens.shape
        
        tok_emb = nn.Embed(self.vocab_size, self.embed_dim)(tokens)
        pos_emb = nn.Embed(self.max_len, self.embed_dim)(jnp.arange(t))
        x = tok_emb + pos_emb[None, :, :]
        
        causal_mask = jnp.tril(jnp.ones((t, t))) == 1
        pad_mask    = (tokens != pad_id)[:, None, None, :]
        mask        = causal_mask[None, None, :, :] & pad_mask

        # INTERNAL RL COMPONENTS INITIALIZATION
        # Моделийн параметр дотор meta_controller болон steering_proj
        # үүсэж байгаа эсэхийг баталгаажуулах dummy call.
        _ = MetaController(latent_dim, name="meta_controller")(x)
        steering_proj = nn.Dense(self.embed_dim, name="steering_proj")
        if intervention_latents is not None:
             _ = steering_proj(jnp.zeros((1, 1, intervention_latents.shape[-1])))

        for i in range(self.num_layers):
            # Controller-т оролт өгөх үед энд зогсоно
            if stop_at_layer is not None and i == stop_at_layer:
                return x

            x = Block(self.embed_dim, self.num_heads)(x, mask)
            
            # INTERVENTION MECHANISM
            # Хэрэв latent vector ирвэл, түүнийг projection хийгээд
            # Residual Stream дээр нэмнэ.
            if i == intervention_layer and intervention_latents is not None:
                steering = jnp.tanh(intervention_latents) 
                # Projection хийх (Latent Dim -> Embed Dim)
                steering_vec = steering_proj(steering)
                x = x + steering_vec

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits


# INITIALIZATION

model = HierarchicalTransformer(vocab_size, embed_dim, num_layers, num_heads, model_max_len)
dummy_in = jnp.zeros((1, sft_seq_len), dtype=jnp.int32)
# Initialization үед dummy latent өгч параметрүүдийг бүрэн үүсгэнэ
dummy_z  = jnp.zeros((1, sft_seq_len, latent_dim))
key      = jax.random.PRNGKey(seed)
params   = model.init(key, dummy_in, intervention_latents=dummy_z)["params"]

print("Model Params Keys:", params.keys())


# JAX HELPERS

def partition_params(params):
    # Аль параметрийг сургах, алийг нь царцаахыг ялгана
    flat_params = flax.traverse_util.flatten_dict(params)
    labels = {}
    for k, v in flat_params.items():
        if 'meta_controller' in k or 'steering_proj' in k:
            labels[k] = 'trainable' # Зөвхөн controller сурна
        else:
            labels[k] = 'frozen'    # Base model царцана
    return flax.traverse_util.unflatten_dict(labels)

@jax.jit
def get_gaussian_logprob(x, mu, log_std):
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(((x - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)

@jax.jit
def generate_rollout_hierarchical(params, prompt_tokens, key, temperature):
    """
    TEMPORAL ABSTRACTION GENERATION
    Токен бүр дээр биш, K алхам тутамд шинэ Latent Z үүсгэж түүнийгээ хадгалан ашиглана.
    """
    B, P = prompt_tokens.shape
    current_seq = jnp.pad(prompt_tokens, ((0,0), (0, gen_len)), constant_values=pad_id)
    
    # Latent Z болон Timer-ийг эхлүүлэх
    initial_z = jnp.zeros((B, latent_dim))
    # Timer: Sequence бүрд тусдаа тоологч (Vector)
    initial_timer = jnp.zeros(B, dtype=jnp.int32) 
    
    init_carry = (
        current_seq,
        key,
        initial_z,
        initial_timer,
        jnp.zeros((gen_len, B)) # Latent Action-уудын Logprob (Token logprob биш)
    )

    def scan_body(carry, i):
        seq, k, curr_z, timer, stored_lps = carry
        
        # Шинэ Z авах цаг болсон эсэх
        is_refresh_step = (timer == 0)
        
        # Base Model-ийн тухайн үеийн hidden state-ийг авах
        hidden_states = model.apply({'params': params}, seq, stop_at_layer=intervention_layer)
        curr_idx      = P + i - 1
        last_hidden   = hidden_states[jnp.arange(B), curr_idx, :]
        
        k, k_gen, k_z = jax.random.split(k, 3)
        
        # MetaController ажиллуулах
        mu, log_std = MetaController(latent_dim, name="meta_controller").apply(
            {'params': params['meta_controller']}, 
            last_hidden
        )
        std = jnp.exp(log_std)
        # Шинэ Z дээжлэх
        new_sample_z = mu + std * jax.random.normal(k_z, std.shape)
        # Энэ action-ий logprob-ийг тооцооллох (Z)
        new_lp_z = get_gaussian_logprob(new_sample_z, mu, log_std)
        
        # Persistence Logic (Hold эсвэл Update)
        active_z = jnp.where(is_refresh_step[:, None], new_sample_z, curr_z)
        # Шинэчлэгдсэн бол timer=k-1, үгүй бол timer-1
        new_timer = jnp.where(is_refresh_step, temporal_k - 1, timer - 1)
        
        # Logprob-ийг зөвхөн refresh хийсэн алхам дээр хадгална, бусад үед 0
        active_lp = jnp.where(is_refresh_step, new_lp_z, 0.0)
        
        # Intervention-тэй Forward Pass
        # Одоогийн алхамд зориулж Z-ийг broadcast хийх
        z_seq_dummy = jnp.zeros((B, seq.shape[1], latent_dim))
        z_seq_dummy = z_seq_dummy.at[:, :, :].set(active_z[:, None, :])
        
        logits = model.apply({'params': params}, seq, intervention_latents=z_seq_dummy)
        
        # Token Sampling
        pred_logits = logits[:, curr_idx, :]
        pred_logits = pred_logits.at[:, pad_id].set(-1e9)
        scaled_logs = pred_logits / temperature
        next_tok    = jax.random.categorical(k_gen, scaled_logs).astype(jnp.int32)
        
        write_idx   = P + i
        new_seq     = seq.at[:, write_idx].set(next_tok)
        new_stored_lps = stored_lps.at[i].set(active_lp)
        
        # Token биш Latent Z-ийг буцаах (Сургалтад зориулж)
        return (new_seq, k, active_z, new_timer, new_stored_lps), (active_z, active_lp)

    (final_seq, _, _, _, all_lps_array), (all_zs, _) = jax.lax.scan(scan_body, init_carry, jnp.arange(gen_len))
    
    # Latent-уудыг буцаах (Batch, Time, Dim)
    all_zs = jnp.swapaxes(all_zs, 0, 1)
    # Logprobs (Batch, Time)
    all_lps_array = jnp.swapaxes(all_lps_array, 0, 1)
    
    return final_seq, all_zs, all_lps_array


# REWARD FUNCTION

def reward_hybrid(text):
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 5: return -2.0
    score = 0.0
    for w in words:
        if w in happy_vocab: score += 2.0
        elif w in sad_vocab: score -= 2.0
    return max(-5.0, min(5.0, score))


def compute_grpo_advantages(rewards, n_prompts, g_size):
    rg   = rewards.reshape(n_prompts, g_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    std  = np.std(rg, axis=1, keepdims=True) + 1e-6
    adv  = (rg - mean) / std # GRPO normalization
    return adv.reshape(-1).astype(np.float32)


# PHASE 1: SFT

sft_tx    = optax.adamw(sft_learning_rate)
sft_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)

@jax.jit
def sft_step(state, batch):
    def loss_fn(p):
        logits = state.apply_fn({'params': p}, batch)
        loss   = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], batch[:, 1:])
        mask   = (batch[:, 1:] != pad_id)
        return jnp.sum(loss * mask) / jnp.sum(mask)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

print("\n" + "="*50)
print("  PHASE 1: SFT - Суурь хэлний мэдлэг олгох")
print("="*50 + "\n")

for step in range(sft_total_steps):
    starts = np.random.randint(0, corpus_ids.shape[0] - sft_seq_len - 1, sft_batch_size)
    batch  = np.stack([corpus_ids[s:s+sft_seq_len+1] for s in starts])
    sft_state, loss = sft_step(sft_state, jnp.asarray(batch))
    
    if step % 500 == 0:
        print(f"[SFT] Step {step:4d} | Loss: {loss:.4f}")

learned_params = sft_state.params
del sft_state, sft_tx
jax.clear_caches()


# PHASE 2: INTERNAL GRPO (Meta-Controller Training)

print("\n" + "="*50)
print("  PHASE 2: INTERNAL GRPO - Meta-Controller сургах")
print("  (Token биш Latent Z вектор дээр сургалт явна)")
print("="*50 + "\n")

# Зөвхөн Controller-ийг сургах Partition
partitioned_params = partition_params(learned_params)
tx_grpo = optax.multi_transform(
    {'trainable': optax.adam(grpo_lr), 'frozen': optax.set_to_zero()},
    partitioned_params
)
grpo_state = train_state.TrainState.create(apply_fn=model.apply, params=learned_params, tx=tx_grpo)
frozen_ref = learned_params # Reference model (Base + Random Controller)

@jax.jit
def internal_grpo_step(state, rollouts, latents, old_lps, advs, beta):
    # Latent Z дээр Loss тооцох функц
    def loss_fn(p):
        B, T_gen, _ = latents.shape
        T_total = rollouts.shape[1]
        prompt_len_curr = T_total - T_gen
        
        # Replay: Base model-ийн hidden state-ийг дахин авах (Энэ удаад бүтэн sequence-ээр нь)
        # Latents-ийг зөв байрлалд нь буцааж хийх (Dummy padding for prompt)
        full_latents = jnp.concatenate([jnp.zeros((B, prompt_len_curr, latent_dim)), latents], axis=1)
        
        # Controller-ийн гаралтууд (mu, log_std) хэрэгтэй тул
        # Loop ашиглах эсвэл customized forward pass хэрэгтэй.
        # Хялбарчлах үүднээс шууд Controller-ийг дуудна.
        
        # Base Model forward (Intervention-гүйгээр hidden авах)
        hidden_states = model.apply({'params': p}, rollouts, stop_at_layer=intervention_layer)
        gen_hidden    = hidden_states[:, prompt_len_curr-1:-1, :] # Input for controller
        
        # Controller forward (re-calculate pi(z|h))
        mu, log_std = MetaController(latent_dim, name="meta_controller").apply(
            {'params': p['meta_controller']}, 
            gen_hidden
        )
        
        # Шинэ Log Prob тооцох
        # latents: (B, T_gen, Dim)
        new_lps = get_gaussian_logprob(latents, mu, log_std) # (B, T_gen)
        
        # Masking: Бид зөвхөн Temporal K алхам тутамд шийдвэр гаргасан.
        # Бусад алхмуудын logprob-ийг тэглэх ёстой (old_lps аль хэдийн 0 байгаа).
        # Гэхдээ old_lps нь 0 байгаа газар mask үүсгэж болно.
        mask = (old_lps != 0.0)
        
        # PPO Loss
        ratio   = jnp.exp(new_lps - old_lps)
        surr1   = ratio * advs[:, None]
        surr2   = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advs[:, None]
        pg_loss = -jnp.sum(jnp.minimum(surr1, surr2) * mask) / jnp.sum(mask)
        
        # KL Divergence (Random Init Controller-тай харьцуулах)
        # (Эсвэл зүгээр л Gaussian Prior руу татах)
        # Internal RL цаасан дээр KL-ийг N(0,1) рүү татдаг (VAE style).
        kl_div  = -0.5 * jnp.sum(1 + log_std - jnp.square(mu) - jnp.exp(log_std), axis=-1)
        kl_loss = jnp.sum(kl_div * mask) / jnp.sum(mask)
        
        return pg_loss + beta * kl_loss, (pg_loss, kl_loss)

    (loss, (pg, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), pg, kl


# GRPO LOOP

kl_coeff = kl_beta
for update in range(grpo_total_updates):
    # Prompt бэлдэх
    p_list = []
    for _ in range(prompts_per_update):
        idx = random.randint(0, len(corpus_ids) - prompt_len - 1)
        p_list.append(corpus_ids[idx : idx + prompt_len])
    prompts = np.repeat(np.stack(p_list), group_size, axis=0)
    
    # Rollout (Hierarchical)
    # Token биш, Latent Z-ийн мэдээллийг (latents, logprobs) авна
    key, k_gen = jax.random.split(key)
    rollouts, latents, latent_lps = generate_rollout_hierarchical(
        grpo_state.params, 
        jnp.asarray(prompts), 
        k_gen, 
        grpo_temp
    )
    
    # Rewards
    text_res = [decode_ids(r[prompt_len:]) for r in rollouts]
    rewards  = np.array([reward_hybrid(t) for t in text_res])
    
    advs = compute_grpo_advantages(rewards, prompts_per_update, group_size)
    
    # Update (Internal Controller)
    # Mini-batch loop-ийг хялбарчлав (бүтнээр нь)
    for _ in range(ppo_epochs):
        grpo_state, pg, kl = internal_grpo_step(
            grpo_state,
            rollouts,
            latents,
            latent_lps,
            jnp.asarray(advs),
            kl_coeff
        )
        
    # Adaptive KL
    if kl > target_kl * 1.5: kl_coeff *= 1.5
    elif kl < target_kl / 1.5: kl_coeff /= 1.5
    
    if update % 20 == 0:
        avg_rew = rewards.mean()
        print(f"[Internal GRPO] Upd {update:4d} | Rew: {avg_rew:.2f} | KL: {kl:.4f} | Out: {text_res[0][:60]}")

print("Done.")