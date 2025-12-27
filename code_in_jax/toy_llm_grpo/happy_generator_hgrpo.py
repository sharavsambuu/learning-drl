#
# HIERARCHICAL LATENT STEERING WITH GRPO (Internal HRL for Autoregressive Models)
#
# https://huggingface.co/papers/2512.20605
# https://arxiv.org/pdf/2512.20605
#
#
# Зорилго
#
# Энэ код нь autoregressive LLM-үүдийн зан авирыг токен, үг бүрийн түвшинд бус харин 
# утга санаа, бодол, зорилгын түвшинд удирдах Hierarchical Reinforcement Learning (HRL)
# аргачлалыг хамгийн энгийн боломжит (MVP) хэлбэрээр хэрэгжүүлэн турших зорилготой.
# Өөрөөр хэлбэл:
#   - Модель текстийг яаж бичихийг Base Transformer шийднэ.
#   - Модель ямар утга, сэтгэл хөдлөл, чиглэл барихыг Meta-Controller (Latent Z) удирдана.
#
#
# Энэхүү хэрэгжүүлэлт нь дараах судалгааны чиглэлүүдийн огтлолцол дээр суурилсан Hybrid архитектур юм
#
# 1. Autoregressive Models as Environments
#    - Pretrained LLM-ийг токен гаргагч биш, харин өөрийн гэсэн дотоод динамиктай орчин гэж үзнэ.
#
# 2. Director / Manager Architectures (DeepMind, Hafner et al.)
#    - Том чадвартай модель (Worker / Body) нь 
#      жижиг стратегийн модель (Manager / Mind)-ээр удирдуулж ажиллана.
#    - Manager нь token биш Latent Goal / Thought вектор ялгаруулна.
#
# 3. Activation Engineering / Steering Vectors
#    - Моделийн жинг өөрчлөхгүйгээр, дунд давхаргын residual stream-д вектор нэмэх замаар зан авирыг чиглүүлнэ.
#
# 4. Temporal Abstraction (Options / Skills)
#    - Шийдвэр гаргалтыг цаг хугацааны хувьд багцална.
#    - Controller нь алхам бүрд биш харин K алхам тутамд нэг удаа утга санаа сонгоно.
#
#
# Архитектурын ерөнхий бүтэц
#
# A. Base Transformer (Body)
#    - Character-level autoregressive Transformer.
#    - Хэлний дүрэм, бүтэц, урсгалыг авч явна.
#    - SFT үе шатанд сургагдаж, HRL үед ихэнхдээ Frozen байна.
#
# B. Meta-Controller (Mind)
#    - Base-ийн дунд давхаргын hidden state-ээс Latent Z (Goal / Thought) вектор үүсгэнэ.
#    - Gaussian policy (mu, log_std) хэлбэртэй.
#
# C. Internal Intervention
#    - Z вектор нь token гаргахгүй.
#    - Residual stream дээр дараах байдлаар нөлөөлдөг:
#         h_modified = h_original + Projection(tanh(Z))
#
# D. Temporal Persistence
#    - Нэг Z нь K алхмын турш хүчинтэй.
#    - Энэ нь модель нэг утга санаа, сэтгэл хөдлөлийг богино хугацаанд тогтвортой барих боломж олгодог.
#
#
# Машин сургалтын үе шатууд
#
# PHASE 1 — SFT (Supervised Fine-Tuning)
#   - Base Transformer нь текстийн суурь чадварыг олж авна.
#   - Үсэг, үг, өгүүлбэрийн зүй тогтлыг сурна.
#
# PHASE 2 — INTERNAL GRPO (Latent HRL / Alignment RL)
#   - Reinforcement Learning нь token дээр биш Latent Z шийдвэрүүд дээр явагдана.
#   - Critic модель ашиглахгүй.
#   - Нэг prompt-ын олон rollout-ыг хооронд нь харьцуулж (Group Relative Policy Optimization)
#     Meta-Controller-ийг сургана.
#
#
# Reward Design
#
#   - Reward нь token-level alignment бус semantic-level зан авирыг дэмжихэд чиглэгдэнэ.
#   - Original GRPO кодын fluency + anti-degeneracy reward логик 
#
#
# Бусад
#
#   Learned termination, multi-layer steering, илүү хүчтэй reward decomposition зэрэг
#   сайжруулалтуудыг нэмж хэрэгжүүлэх боломжтой. 
#
#


import os
# JAX ийн санах ойн хуваарилалтыг хязгаарлах буюу OOM алдаанаас сэргийлэх тохиргоо
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


debug                 = True
dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"
seed                  = 42


# HYPERPARAMETERS, 10GB VRAM зориулсан тохиргоо

# SFT буюу Supervised Fine-Tuning үе шатны тохиргоо
sft_total_steps       = 10000
sft_batch_size        = 128
sft_seq_len           = 256     # Context window
sft_learning_rate     = 5e-4
sft_warmup_steps      = 200
sft_sample_freq       = 500     # Хэдэн алхам тутамд SFT үр дүнг хэвлэж харах вэ

# INTERNAL GRPO буюу Reinforcement Learning үе шатны тохиргоо
grpo_total_updates    = 6500
group_size            = 16      # GRPO ийн харьцуулалт хийх бүлгийн хэмжээ
prompts_per_update    = 4       # Нэг алхамд 4 prompt үржихдэг нь 16 group буюу 64 rollout
gen_len               = 256     # Үүсгэх текстийн урт
grpo_temp             = 0.9     # Текст үүсгэх температур
grpo_sample_freq      = 20      # Хэдэн update тутамд үр дүнг хэвлэж харах вэ

# PPO буюу Proximal Policy Optimization тохиргоо
ppo_epochs            = 3
mini_batch_size       = 16      # 10GB VRAM-д тааруулан Gradient Accumulation хийнэ (64 -> 16)
accum_steps           = 4       # 64 rollout / 16 batch = 4 steps accumulation
clip_epsilon          = 0.2
entropy_coeff         = 0.00    # Controller дээр token-entropy биш, энд 0 байж болно
grpo_lr               = 2e-5    # Controller fine-tuning үед маш бага LR хэрэг болно
max_grad_norm         = 1.0

# Dynamic KL Divergence буюу Controller галзуурахаас сэргийлэх тохиргоо
kl_beta               = 0.05    # Анхны шийтгэлийн хэмжээ
target_kl             = 0.06    # Зорилтот өөрчлөлтийн хэмжээ
kl_alpha              = 1.2     # Beta-г өөрчлөх хурд

# Internal Control Parameters
latent_dim            = 32      # Удирдлагын векторын хэмжээ
temporal_k            = 4       # Нэг бодлыг хэдэн алхам барих вэ (Persistence)
intervention_layer    = 1       # Аль давхаргад нөлөөлөх вэ (0..num_layers-1)

# Моделийн дотоод бүтцийн тохиргоо
prompt_len            = 48
model_max_len         = 320     # Positional Embedding-д хэрэгтэй
num_layers            = 4       # Transformer блокуудын тоо
num_heads             = 4       # Multi-head attention heads
embed_dim             = 128     # Embedding size
dropout_rate          = 0.1     # Overfitting-ээс сэргийлнэ


# Эерэг болон сөрөг үгсийн сан
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

# Түүхүүдийг задлах буюу SFT сургалтад бэлдэх
all_stories = [s.strip() for s in raw_text.split(end_of_text_token) if len(s.strip()) > 0]
# GRPO шатанд ашиглах эерэг түүхүүдийн дээж буюу reference болгож ашиглана
positive_stories = [s for s in all_stories if any(w in s.lower() for w in happy_vocab)]
if not positive_stories: positive_stories = all_stories

# Тэмдэгт болон ID хөрвүүлэлт хийх Character Level Tokenizer
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

corpus_text = "\n\n".join(positive_stories)
corpus_ids  = np.array(encode_text(corpus_text), dtype=np.int32)

print(f"Vocab Size  : {vocab_size}")
print(f"Total Chars : {len(corpus_ids)}")



# MODEL ARCHITECTURE, TinyTransformer + Meta-Controller + Steering

class MetaController(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, h):
        # Hidden state -> Latent Distribution (Gaussian)
        x       = nn.Dense(64)(h)
        x       = nn.relu(x)
        mu      = nn.Dense(self.latent_dim)(x)
        log_std = nn.Dense(self.latent_dim)(x)
        # Numerical stability
        log_std = jnp.clip(log_std, -5, 2)
        return mu, log_std

class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Einsum ашиглахгүйгээр энгийн аргаар (Reshape -> Transpose -> Matmul)
        B, T, C  = x.shape
        head_dim = self.embed_dim // self.num_heads

        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(x)
        v = nn.Dense(self.embed_dim)(x)

        # Split heads: [Batch, Time, Heads, HeadDim] -> [Batch, Heads, Time, HeadDim]
        q = q.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention score: Q @ K.T -> [Batch, Heads, Time, Time]
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)

        # Combined Masking (Causal + Padding)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Weighted sum: Attn @ V -> [Batch, Heads, Time, HeadDim]
        out = jnp.matmul(attn_weights, v)

        # Reassemble: [Batch, Time, Heads, HeadDim] -> [Batch, Time, EmbedDim]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)

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
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Attention with Residual
        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(nn.LayerNorm()(x), mask, deterministic)
        # MLP with Residual
        x = x + MLP(self.embed_dim)(nn.LayerNorm()(x))
        return x

class HierarchicalTinyTransformer(nn.Module):
    vocab_size : int
    embed_dim  : int
    num_layers : int
    num_heads  : int
    max_len    : int
    latent_dim : int

    @nn.compact
    def __call__(
        self,
        tokens,
        deterministic = True,
        stop_at_layer = None,
        steer_z       = None,
        steer_pos     = None,
    ):
        b, t = tokens.shape

        # Token & Positional Embeddings
        tok_emb = nn.Embed(self.vocab_size, self.embed_dim)(tokens)
        pos_emb = nn.Embed(self.max_len, self.embed_dim)(jnp.arange(t))
        x       = tok_emb + pos_emb[None, :, :]

        # Mask үүсгэх : Causal Mask болон Padding Mask
        causal_mask = jnp.tril(jnp.ones((t, t))) == 1
        pad_mask    = (tokens != pad_id)[:, None, None, :]
        mask        = causal_mask[None, None, :, :] & pad_mask

        # Controller + Steering Projection
        meta_controller = MetaController(self.latent_dim, name="meta_controller")
        steering_proj   = nn.Dense(self.embed_dim, name="steering_proj")

        # Meta-Controller parameter-уудыг init үеэс эхлэн үүсгэхийн тулд dummy forward
        _ = meta_controller(x[:, 0, :])

        # Steering projection parameter-уудыг init үеэс эхлэн үүсгэхийн тулд dummy forward
        _ = steering_proj(jnp.zeros((b, self.latent_dim), dtype=x.dtype))

        for i in range(self.num_layers):
            # Stop-at-layer: hidden_states авахад ашиглана
            if stop_at_layer is not None and i == stop_at_layer:
                return x

            x = Block(self.embed_dim, self.num_heads)(x, mask, deterministic)

            # INTERVENTION MECHANISM
            # Зөвхөн тухайн timestep-ийн position дээр нэмнэ (broadcast хийхгүй)
            if (i == intervention_layer) and (steer_z is not None) and (steer_pos is not None):
                steering  = jnp.tanh(steer_z)
                steer_vec = steering_proj(steering)  # [B, embed_dim]
                x         = x.at[:, steer_pos, :].add(steer_vec)

        x      = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits


# Моделийг цэнэглэн эхлүүлэх
model    = HierarchicalTinyTransformer(vocab_size, embed_dim, num_layers, num_heads, model_max_len, latent_dim)
dummy_in = jnp.zeros((1, sft_seq_len), dtype=jnp.int32)
params   = model.init(jax.random.PRNGKey(seed), dummy_in, deterministic=True)["params"]


# JAX HELPERS

@jax.jit
def logprob_from_logits(logits, actions):
    logp          = jax.nn.log_softmax(logits, axis=-1)
    selected_logp = jnp.take_along_axis(logp, actions[..., None], axis=-1)
    return selected_logp.squeeze(-1)

@jax.jit
def gaussian_logprob(z, mu, log_std):
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(((z - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)

@jax.jit
def gaussian_kl_diag(mu_q, log_std_q, mu_p, log_std_p):
    # KL( N(mu_q, std_q) || N(mu_p, std_p) ) diagonal Gaussians
    std_q = jnp.exp(log_std_q)
    std_p = jnp.exp(log_std_p)
    return jnp.sum(
        log_std_p - log_std_q + (std_q**2 + (mu_q - mu_p)**2) / (2.0 * (std_p**2) + 1e-8) - 0.5,
        axis=-1
    )

@jax.jit
def unroll_logits_eval(p, token_seq):
    return model.apply({"params": p}, token_seq, deterministic=True)

@jax.jit
def unroll_hidden_stop(p, token_seq):
    # intervention_layer дээрх hidden state-г авна (stop_at_layer)
    return model.apply({"params": p}, token_seq, deterministic=True, stop_at_layer=intervention_layer)

def partition_params(p):
    # Аль параметрийг сургах, алийг нь царцаахыг ялгана
    flat_params = flax.traverse_util.flatten_dict(p)
    labels = {}
    for k, v in flat_params.items():
        is_trainable = False
        for kk in k:
            if kk == "meta_controller" or kk == "steering_proj":
                is_trainable = True
                break
        labels[k] = "trainable" if is_trainable else "frozen"
    return flax.traverse_util.unflatten_dict(labels)


@jax.jit
def generate_rollout_hierarchical(behavior_params, prompt_tokens, key, temperature):
    """
    TEMPORAL ABSTRACTION GENERATION
    Токен бүр дээр биш, K алхам тутамд шинэ Latent Z үүсгэж түүнийгээ хадгалан ашиглана.
    Мөн steering нь зөвхөн тухайн timestep-ийн position дээр (pred_idx) нэмэгдэнэ.
    """
    B, P        = prompt_tokens.shape
    current_seq = jnp.pad(prompt_tokens, ((0,0), (0, gen_len)), constant_values=pad_id)

    # Latent Z болон Timer-ийг эхлүүлэх
    initial_z     = jnp.zeros((B, latent_dim))
    initial_timer = jnp.zeros((B,), dtype=jnp.int32)

    init_carry = (current_seq, key, initial_z, initial_timer)

    def scan_body(carry, i):
        seq, k, curr_z, timer = carry

        # refresh time болсон эсэх
        is_refresh_step = (timer == 0)

        # Base Model-ийн hidden state-ийг авах (stop_at_layer)
        hidden_states = unroll_hidden_stop(behavior_params, seq)
        pred_idx      = P + i - 1
        last_hidden   = hidden_states[jnp.arange(B), pred_idx, :]

        # Controller ажиллуулах
        k, k_tok, k_z = jax.random.split(k, 3)

        mu, log_std = MetaController(latent_dim, name="meta_controller").apply(
            {"params": behavior_params["meta_controller"]},
            last_hidden
        )

        new_sample_z = mu + jnp.exp(log_std) * jax.random.normal(k_z, mu.shape)
        new_lp_z     = gaussian_logprob(new_sample_z, mu, log_std)

        # Persistence Logic
        active_z   = jnp.where(is_refresh_step[:, None], new_sample_z, curr_z)
        new_timer  = jnp.where(is_refresh_step, temporal_k - 1, timer - 1)

        # Decision mask + logprob зөвхөн refresh дээр хадгална
        decision_mask = is_refresh_step
        active_lp     = jnp.where(decision_mask, new_lp_z, 0.0)

        # Token sampling хийх logits (Intervention-тэй)
        logits = model.apply(
            {"params": behavior_params},
            seq,
            deterministic = True,
            steer_z       = active_z,
            steer_pos     = pred_idx
        )

        pred_logits   = logits[:, pred_idx, :]
        pred_logits   = pred_logits.at[:, pad_id].set(-1e9)
        scaled_logits = pred_logits / temperature

        next_tok  = jax.random.categorical(k_tok, scaled_logits).astype(jnp.int32)

        write_idx = P + i
        seq       = seq.at[:, write_idx].set(next_tok)

        return (seq, k, active_z, new_timer), (active_z, active_lp, decision_mask)

    (final_seq, _, _, _), (all_zs, all_lps, all_masks) = jax.lax.scan(
        scan_body,
        init_carry,
        jnp.arange(gen_len)
    )

    all_zs    = jnp.swapaxes(all_zs,    0, 1)
    all_lps   = jnp.swapaxes(all_lps,   0, 1)
    all_masks = jnp.swapaxes(all_masks, 0, 1)

    return final_seq, all_zs, all_lps, all_masks


# REWARD FUNCTION, Rule-based

def reward_hybrid_pro(text, fluency_score):
    t     = text.lower()
    words = re.findall(r"[a-z']+", t)

    if len(words) < 6: return -4.0

    score         = 0.0
    happy_matches = 0

    for i, w in enumerate(words):
        if w in happy_vocab:
            context = words[max(0, i-2):i]
            if any(n in context for n in negations):
                score -= 3.0
            else:
                score         += 2.5
                happy_matches += 1
        elif w in sad_vocab:
            score -= 2.0

    for w in set(words):
        count = words.count(w)
        if count > 3: score -= (count - 3) * 1.0

    diversity = len(set(words)) / len(words)
    score    += diversity * 4.0

    if fluency_score < -3.5: score -= 4.0

    if text.strip().endswith(('.', '!', '?')): score += 1.5
    if len(text) > 100: score += 1.0

    return max(-10.0, min(10.0, score))


def compute_grpo_advantages(rewards, n_prompts, g_size):
    """
    GRPO ийн гол логик буюу Mean-Centric Advantage
    STD хуваахгүй байх нь бага бүлэгт (Group Size=16) илүү тогтвортой
    """
    rg   = rewards.reshape(n_prompts, g_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    adv  = (rg - mean)
    adv  = np.clip(adv, -5.0, 5.0)
    return adv.reshape(-1).astype(np.float32), float(mean.mean())



# PHASE 1, SFT (Supervised Fine-Tuning)

sft_tx    = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adamw(
        optax.warmup_cosine_decay_schedule(0, sft_learning_rate, sft_warmup_steps, sft_total_steps),
        weight_decay=1e-4
    )
)
sft_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)

@jax.jit
def sft_step(state, batch):
    def loss_fn(p):
        logits       = unroll_logits_eval(p, batch)
        logits_trunc = logits[:, :-1, :]
        labels       = batch[:, 1:]
        mask         = (labels != pad_id).astype(jnp.float32)
        loss         = optax.softmax_cross_entropy_with_integer_labels(logits_trunc, labels)
        return jnp.sum(loss * mask) / jnp.sum(mask)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

print("\n" + "="*50)
print("  PHASE 1: SFT - Суурь хэлний мэдлэг олгох (Transformer)")
print(f"  Steps: {sft_total_steps} | Batch: {sft_batch_size} | Device: GPU (JIT)")
print("="*50 + "\n")

for step in range(sft_total_steps):
    starts = np.random.randint(0, corpus_ids.shape[0] - sft_seq_len - 1, sft_batch_size)
    batch  = np.stack([corpus_ids[s:s+sft_seq_len+1] for s in starts])

    sft_state, sft_loss = sft_step(sft_state, jnp.asarray(batch))

    if step % 500 == 0:
        print(f"[SFT] Step {step:4d} | Loss: {sft_loss:.4f}")

    if step > 0 and step % sft_sample_freq == 0:
        test_key    = jax.random.PRNGKey(step)
        test_prompt = jnp.full((1, prompt_len), pad_id, dtype=jnp.int32)
        start_phr   = encode_text("Once upon a time")[:prompt_len]
        test_prompt = test_prompt.at[0, :len(start_phr)].set(jnp.array(start_phr))

        sample, _, _, _ = generate_rollout_hierarchical(sft_state.params, test_prompt, test_key, 0.8)
        decoded     = decode_ids(sample[0])
        print(f"   >> Sample: {decoded[len(decode_ids(test_prompt[0])):][:120]}...")

# SFT-ийн үр дүнг хадгалаад санах ойг цэвэрлэх
print("\n[Memory] Cleaning SFT states...")
learned_params = sft_state.params
del sft_state, sft_tx
jax.clear_caches()
gc.collect()


# PHASE 2, INTERNAL GRPO (Meta-Controller Training)

# Зөвхөн Controller-ийг сургах Partition
partitioned_params = partition_params(learned_params)
tx_grpo = optax.multi_transform(
    {
        "trainable": optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(grpo_lr)),
        "frozen":    optax.set_to_zero()
    },
    partitioned_params
)

grpo_state = train_state.TrainState.create(apply_fn=model.apply, params=learned_params, tx=tx_grpo)

# Reference policy (frozen copy)
frozen_ref = learned_params

@jax.jit
def internal_grpo_step_minibatch(state, ref_params, rollouts, z_seq, old_lps, decision_mask, advs, beta):
    """
    Controller дээр PPO/GRPO update хийх (Gradient Accumulation)
    Token биш Latent Z decision дээр ratio тооцно.
    """
    r_rollouts = rollouts     .reshape(accum_steps, mini_batch_size, -1)
    r_zseq     = z_seq        .reshape(accum_steps, mini_batch_size, -1, latent_dim)
    r_oldlps   = old_lps      .reshape(accum_steps, mini_batch_size, -1)
    r_mask     = decision_mask.reshape(accum_steps, mini_batch_size, -1)
    r_advs     = advs         .reshape(accum_steps, mini_batch_size)

    gen_start  = prompt_len

    def compute_grad(carry, i):
        curr_st = carry
        b_roll  = r_rollouts[i]
        b_zseq  = r_zseq    [i]
        b_oldlp = r_oldlps  [i]
        b_msk   = r_mask    [i]
        b_adv   = r_advs    [i]

        def loss_fn(p):
            hidden_all = unroll_hidden_stop(p, b_roll)
            gen_hidden = hidden_all[:, gen_start-1:gen_start+gen_len-1, :]

            mu, log_std = MetaController(latent_dim, name="meta_controller").apply(
                {"params": p["meta_controller"]},
                gen_hidden
            )

            mu_ref, log_std_ref = MetaController(latent_dim, name="meta_controller").apply(
                {"params": ref_params["meta_controller"]},
                gen_hidden
            )

            new_lps  = gaussian_logprob(b_zseq, mu, log_std)

            m        = b_msk.astype(jnp.bool_)

            safe_new = jnp.where(m, new_lps, b_oldlp)
            safe_old = jnp.where(m, b_oldlp, safe_new)

            ratio    = jnp.exp(jnp.clip(safe_new - safe_old, -20.0, 20.0))
            surr1    = ratio * b_adv[:, None]
            surr2    = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_adv[:, None]

            denom    = jnp.maximum(jnp.sum(m), 1.0)
            pg_loss  = -jnp.sum(jnp.minimum(surr1, surr2) * m) / denom

            kl_each  = gaussian_kl_diag(mu, log_std, mu_ref, log_std_ref)
            kl_loss  = jnp.sum(kl_each * m) / denom

            total    = pg_loss + beta * kl_loss
            return total, (pg_loss, kl_loss)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(curr_st.params)
        return curr_st, (grads, loss, aux)

    _, (all_grads, all_losses, all_aux) = jax.lax.scan(compute_grad, state, jnp.arange(accum_steps))

    grads_avg = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0) / accum_steps, all_grads)
    avg_loss  = jnp.mean(all_losses)
    avg_pg    = jnp.mean(all_aux[0])
    avg_kl    = jnp.mean(all_aux[1])

    return state.apply_gradients(grads=grads_avg), avg_loss, avg_pg, avg_kl


print("\n" + "="*50)
print("  PHASE 2: INTERNAL GRPO - Meta-Controller сургах (Temporal Abstraction)")
print(f"  Updates: {grpo_total_updates} | Group: {group_size} | MiniBatch: {mini_batch_size} | K: {temporal_k}")
print("="*50 + "\n")

kl_coeff = kl_beta

for update in range(grpo_total_updates):
    # Санамсаргүй prompt бэлдэх
    p_list = []
    for _ in range(prompts_per_update):
        story              = random.choice(all_stories)
        ids                = encode_text(story)
        start              = random.randint(0, max(0, len(ids) - prompt_len))
        p_ids              = np.full((prompt_len,), pad_id, dtype=np.int32)
        chunk              = ids[start:start+prompt_len]
        p_ids[:len(chunk)] = chunk
        p_list.append(p_ids)

    prompts = np.repeat(np.stack(p_list), group_size, axis=0)

    # Rollout (Hierarchical)
    key = jax.random.PRNGKey(update)
    rollouts, z_seq, z_lps, dmask = generate_rollout_hierarchical(
        grpo_state.params,
        jnp.asarray(prompts),
        key,
        grpo_temp
    )

    # Fluency Score & Rewards 
    ref_logits_all  = unroll_logits_eval(frozen_ref, rollouts)
    ref_gen_logits  = ref_logits_all[:, prompt_len-1:-1, :]
    target_lps      = logprob_from_logits(ref_gen_logits, rollouts[:, prompt_len:])
    fluency         = np.array(jnp.mean(target_lps, axis=1))

    rewards = np.array([
        reward_hybrid_pro(decode_ids(rollouts[i, prompt_len:]), fluency[i])
        for i in range(rollouts.shape[0])
    ])

    advs, m_reward = compute_grpo_advantages(rewards, prompts_per_update, group_size)

    # PPO Update (mini-batch, accumulation)
    for _ in range(ppo_epochs):
        perm = np.random.permutation(prompts.shape[0])

        grpo_state, _, pg_l, kl_l = internal_grpo_step_minibatch(
            grpo_state,
            frozen_ref,
            rollouts[perm],
            z_seq[perm],
            z_lps[perm],
            dmask[perm],
            jnp.asarray(advs[perm]),
            kl_coeff
        )

    # Dynamic KL Controller
    kl_val = float(kl_l)
    if kl_val > target_kl * 1.5:
        kl_coeff *= kl_alpha
    elif kl_val < target_kl / 1.5:
        kl_coeff /= kl_alpha

    # Явц хэвлэж харах
    if update % 20 == 0:
        print(f"[Internal GRPO] Upd {update:4d} | AvgReward: {m_reward:6.2f} | KL: {kl_val:.4f} | Beta: {kl_coeff:.4f}")

    if update % grpo_sample_freq == 0:
        best_idx = np.argmax(rewards)
        best_txt = decode_ids(rollouts[best_idx, prompt_len:])
        print(f"   >> Best sample : {best_txt[:160]}...")


print("\n=== СУРГАЛТ ДУУСЛАА ===")
final_prompt        = np.full((1, prompt_len), pad_id, dtype=np.int32)
final_prompt[0, :4] = encode_text("Once")[:4]
final_rollout, _, _, _ = generate_rollout_hierarchical(grpo_state.params, jnp.asarray(final_prompt), jax.random.PRNGKey(999), 0.8)
print(f"Сүүлийн үр дүн: {decode_ids(final_rollout[0, prompt_len:])}")
