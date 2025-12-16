#
# HAPPY TEXT GENERATOR (Toy LLM)
# Critic-less GRPO (Group Relative Policy Optimization)
#
# Architecture:
#   - Char-level LSTM + Tiny Causal Attention (Rolling Window)
#   - Critic-less PPO with Group Relative Advantages
#
# Training Phases:
#   1. SFT (Supervised Fine-Tuning):
#      - Bias-filtered dataset (positive_stories)
#      - Learns grammar and happy vocabulary
#
#   2. GRPO (Alignment):
#      - Samples multiple completions per prompt
#      - Hybrid Reward: Lexical (Happy Words) + Structure (N-grams) + Fluency (Ref Model)
#      - Masked Advantages: Ignores "best of worst" negative outcomes
#      - Prompt Mixing: Blends positive/random prompts to ensure robustness
#      - Frozen Reference Model: Prevents drift
#

import os
# prevent jax from pre-allocating all vram
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"
import re
import math
import random
import jax
import optax
import numpy         as np
import flax.linen    as nn
from   jax           import numpy as jnp
from   flax.training import train_state


debug                 = True
debug_render          = True

dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"

seed                  = 42

# phase 1: sft configs 
sft_total_steps       = 10000
sft_batch_size        = 128
sft_seq_len           = 192
sft_learning_rate     = 0.0015
sft_warmup_steps      = 2000
sft_print_freq        = 1000
sft_print_count       = 2

# phase 2: grpo configs 
grpo_total_updates    = 2000
prompts_per_update    = 16
group_size            = 8
gen_len               = 64
grpo_temp             = 1.0    # set to 1.0 to match ppo logratio expectations

ppo_epochs            = 4
mini_batch_size       = 128
clip_epsilon          = 0.2
entropy_coeff         = 0.02   
kl_beta               = 0.04
grpo_lr               = 0.0005
max_grad_norm         = 0.5
use_std_advantage     = True

grpo_print_freq       = 50
grpo_print_count      = 2

# prompt mixing schedule 
mix_stage1_updates    = 500
mix_stage2_updates    = 1500

# stability & reward knobs 
adv_clip_max          = 5.0
reward_clip_max       = 10.0
fluency_threshold     = -2.5

# model params
min_prompt_tokens     = 2
prompt_len            = 32
num_lstm_layers       = 2
dropout_rate          = 0.10
attn_window           = 16
embed_dim             = 64
hidden_dim            = 256

# lexical definitions
happy_vocab           = [
    "happy","joy","joyful","smile","smiled","laugh","laughed","love","loved","kind","nice","fun",
    "good","great","amazing","wonderful","excited","brave","bright","safe","friend","friends"
]
sad_vocab             = [
    "sad","cry","cried","bad","angry","mad","hurt","scary","afraid","fear","dark",
    "hate","hated","mean","alone","lost","dead","death","kill","killed"
]


np.random.seed(seed)
random.seed(seed)


if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

# split raw text into individual stories
all_stories = [s.strip() for s in raw_text.split(end_of_text_token)]
all_stories = [s for s in all_stories if len(s) > 0]

# filter: keep only stories containing at least one happy word
positive_stories = []
for s in all_stories:
    s_lower = s.lower()
    if any(w in s_lower for w in happy_vocab):
        positive_stories.append(s)

if len(positive_stories) < 100:
    print("warning: filter too aggressive, falling back to full dataset")
    positive_stories = all_stories

# build char-level vocab from full dataset
unique_chars = set()
for s in all_stories:
    for ch in s:
        unique_chars.add(ch)

PAD = "<PAD>"
BOS = "<BOS>"

chars = [PAD, BOS] + sorted(list(unique_chars))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}

vocab_size  = len(chars)
pad_id      = char_to_id[PAD]
bos_id      = char_to_id[BOS]

fallback_char = " "
if fallback_char not in char_to_id:
    fallback_char = "\n" if "\n" in char_to_id else None
fallback_id = char_to_id[fallback_char] if fallback_char is not None else bos_id


def encode_text(text):
    ids = [bos_id]
    for ch in text:
        ids.append(char_to_id.get(ch, pad_id))
    return ids


def decode_ids(ids):
    out = []
    for i in ids:
        ii = int(i)
        if ii == pad_id: continue
        token = id_to_char[ii]
        if token == BOS: continue
        out.append(token)
    return "".join(out)


corpus_text = "\n\n".join(positive_stories)
corpus_ids  = np.array(encode_text(corpus_text), dtype=np.int32)


class TinyLSTM(nn.Module):
    vocab_size  : int
    embed_dim   : int
    hidden_dim  : int
    num_layers  : int
    dropout     : float
    attn_window : int

    @nn.compact
    def __call__(self, token_id, carry, deterministic=True):
        # carry: (lstm_carries, attn_memory)
        lstm_carries, attn_mem = carry

        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(token_id)

        new_carries = []
        for i in range(self.num_layers):
            c_i, h_i      = lstm_carries[i]
            (c_i, h_i), h = nn.LSTMCell(features=self.hidden_dim, name=f"lstm_{i}")((c_i, h_i), x)

            if i < (self.num_layers - 1):
                h = nn.Dropout(rate=self.dropout, name=f"drop_{i}")(h, deterministic=deterministic)

            new_carries.append((c_i, h))
            x = h

        h_last = x

        # rolling attention memory update
        attn_mem = jnp.concatenate([attn_mem[:, 1:, :], h_last[:, None, :]], axis=1)

        # tiny causal attention
        q = nn.Dense(features=self.hidden_dim, name="attn_q")(h_last)
        k = nn.Dense(features=self.hidden_dim, name="attn_k")(attn_mem)
        v = nn.Dense(features=self.hidden_dim, name="attn_v")(attn_mem)

        scale  = 1.0 / jnp.sqrt(jnp.array(self.hidden_dim, dtype=jnp.float32))
        scores = jnp.sum(k * q[:, None, :], axis=-1) * scale
        w      = jax.nn.softmax(scores, axis=-1)
        ctx    = jnp.sum(v * w[:, :, None], axis=1)

        h_cat  = jnp.concatenate([h_last, ctx], axis=-1)
        h_mix  = nn.Dense(features=self.hidden_dim, name="attn_mix")(h_cat)
        h_mix  = nn.tanh(h_mix)
        h_mix  = nn.LayerNorm(name="ln_out")(h_mix)
        h_mix  = nn.Dropout(rate=self.dropout, name="drop_out")(h_mix, deterministic=deterministic)

        logits = nn.Dense(features=self.vocab_size)(h_mix)

        new_carry = (tuple(new_carries), attn_mem)
        return logits, new_carry


def init_carry(batch_size):
    carries = []
    for _ in range(num_lstm_layers):
        c = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
        h = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
        carries.append((c, h))

    attn_mem = jnp.zeros((batch_size, attn_window, hidden_dim), dtype=jnp.float32)
    return (tuple(carries), attn_mem)


model = TinyLSTM(
    vocab_size  = vocab_size     ,
    embed_dim   = embed_dim      ,
    hidden_dim  = hidden_dim     ,
    num_layers  = num_lstm_layers,
    dropout     = dropout_rate   ,
    attn_window = attn_window
)

dummy_token = jnp.zeros((1,), dtype=jnp.int32)
dummy_carry = init_carry(1)
params      = model.init(jax.random.PRNGKey(seed), dummy_token, dummy_carry, deterministic=True)["params"]


@jax.jit
def logprob_from_logits(logits, actions):
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp_all, actions[:, None], axis=1).squeeze(1)


@jax.jit
def entropy_from_logits(logits_2d):
    p  = jax.nn.softmax(logits_2d, axis=-1)
    lp = jax.nn.log_softmax(logits_2d, axis=-1)
    return -jnp.sum(p * lp, axis=-1).mean()


@jax.jit
def kl_from_logits(logits_new_2d, logits_ref_2d):
    p_new  = jax.nn.softmax(logits_new_2d, axis=-1)
    lp_new = jax.nn.log_softmax(logits_new_2d, axis=-1)
    lp_ref = jax.nn.log_softmax(logits_ref_2d, axis=-1)
    return jnp.sum(p_new * (lp_new - lp_ref), axis=-1).mean()


@jax.jit
def unroll_logits_eval(params, token_seq):
    B, T  = token_seq.shape
    carry = init_carry(B)

    def step_fn(carry, tok):
        logits, carry = model.apply({"params": params}, tok, carry, deterministic=True)
        return carry, logits

    carry, logits_seq = jax.lax.scan(step_fn, carry, token_seq.T)
    logits_seq        = jnp.transpose(logits_seq, (1, 0, 2))
    return logits_seq


@jax.jit
def unroll_logits_train(params, token_seq, key):
    B, T  = token_seq.shape
    carry = init_carry(B)
    keys  = jax.random.split(key, T)

    def step_fn(carry, xs):
        tok, k = xs
        logits, carry = model.apply({"params": params}, tok, carry, deterministic=False, rngs={"dropout": k})
        return carry, logits

    carry, logits_seq = jax.lax.scan(step_fn, carry, (token_seq.T, keys))
    logits_seq        = jnp.transpose(logits_seq, (1, 0, 2))
    return logits_seq


@jax.jit
def condition_on_prompt(params, prompt_tokens):
    B, P  = prompt_tokens.shape
    carry = init_carry(B)

    def step_fn(carry, tok):
        logits, carry = model.apply({"params": params}, tok, carry, deterministic=True)
        return carry, logits

    carry, _ = jax.lax.scan(step_fn, carry, prompt_tokens.T)
    return carry


@jax.jit
def generate_rollout(behavior_params, prompt_tokens, key, temperature):
    N, P     = prompt_tokens.shape
    carry    = condition_on_prompt(behavior_params, prompt_tokens)
    last_tok = prompt_tokens[:, -1]

    def gen_step(carry_key_tok, _):
        carry, key, tok = carry_key_tok
        logits, carry   = model.apply({"params": behavior_params}, tok, carry, deterministic=True)
        
        # temperature scaling (identity if temp=1.0)
        logits          = logits / temperature
        
        key, subkey     = jax.random.split(key)
        next_tok        = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        lp              = logprob_from_logits(logits, next_tok)
        return (carry, key, next_tok), (next_tok, lp)

    (carry, key, _), (gen_toks, gen_lps) = jax.lax.scan(
        gen_step, (carry, key, last_tok), xs=None, length=gen_len
    )

    gen_toks     = jnp.transpose(gen_toks, (1, 0))
    gen_lps      = jnp.transpose(gen_lps , (1, 0))
    rollout_seqs = jnp.concatenate([prompt_tokens, gen_toks], axis=1)
    return rollout_seqs, gen_lps


@jax.jit
def compute_fluency_batch(ref_params, rollout_seqs):
    # check if ref model considers generated text "likely" (fluency gate)
    inputs      = rollout_seqs[:, :-1]
    targets     = rollout_seqs[:,  1:]
    logits      = unroll_logits_eval(ref_params, inputs)
    logp_all    = jax.nn.log_softmax(logits, axis=-1)

    target_logp = jnp.take_along_axis(logp_all, targets[:, :, None], axis=-1).squeeze(-1)

    gen_start   = prompt_len - 1
    gen_end     = gen_start + gen_len
    gen_logps   = target_logp[:, gen_start:gen_end]

    return jnp.mean(gen_logps, axis=1)


sft_lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value   = 0.0,
    peak_value   = sft_learning_rate,
    warmup_steps = sft_warmup_steps,
    decay_steps  = max(1, sft_total_steps - sft_warmup_steps),
    end_value    = sft_learning_rate * 0.1
)

sft_tx = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(sft_lr_schedule)
)
sft_state    = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)


def sample_sft_batch(batch_size, seq_len):
    seq_len  = max(2, int(seq_len))
    need     = int(seq_len + 1)
    N        = int(corpus_ids.shape[0])

    if N <= need + 1:
        tiled = np.tile(corpus_ids, reps=int(math.ceil((need + 2) / max(1, N))))
        max_start = int(tiled.shape[0] - need - 1)
        starts = np.random.randint(0, max_start + 1, size=(batch_size,))
        batch  = np.stack([tiled[s:s + need] for s in starts], axis=0).astype(np.int32)
        return batch

    max_start = int(N - need - 1)
    starts    = np.random.randint(0, max_start + 1, size=(batch_size,))
    batch     = np.stack([corpus_ids[s:s + need] for s in starts], axis=0).astype(np.int32)
    return batch


@jax.jit
def sft_train_step(state, batch_tokens, dropout_key):
    inputs  = batch_tokens[:, :-1]
    targets = batch_tokens[:,  1:]

    def loss_fn(p):
        logits = unroll_logits_train(p, inputs, dropout_key)
        logp   = jax.nn.log_softmax(logits, axis=-1)
        onehot = jax.nn.one_hot(targets, vocab_size)
        nll    = -jnp.sum(logp * onehot, axis=-1)
        mask   = (targets != pad_id).astype(jnp.float32)
        return jnp.sum(nll * mask) / (jnp.sum(mask) + 1e-8)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state       = state.apply_gradients(grads=grads)
    return state, loss


def ensure_valid_prompt(prompt_ids, min_tokens):
    if int(prompt_ids[0]) != bos_id:
        prompt_ids[0] = bos_id
    non_pad = int(np.sum(prompt_ids != pad_id))
    if non_pad >= min_tokens:
        return prompt_ids
    if prompt_len > 1:
        prompt_ids[1] = fallback_id
    return prompt_ids


def sample_random_prompt(source_stories, prompt_len):
    prompt_ids = np.full((prompt_len,), pad_id, dtype=np.int32)
    s          = source_stories[np.random.randint(0, len(source_stories))].replace("\r", "")
    ids        = encode_text(s)

    if len(ids) <= 1:
        prompt_ids[0] = bos_id
        if prompt_len > 1: prompt_ids[1] = fallback_id
        return prompt_ids

    if len(ids) <= prompt_len:
        take = ids[:prompt_len]
    else:
        start_max = max(1, len(ids) - prompt_len)
        start     = np.random.randint(0, start_max)
        take      = ids[start:start + prompt_len]

    take = take[:prompt_len]
    prompt_ids[:len(take)] = np.array(take, dtype=np.int32)
    prompt_ids = ensure_valid_prompt(prompt_ids, min_prompt_tokens)
    return prompt_ids


def sample_snapshot_from(params, n_samples, source_stories):
    prompt_ids_1d = sample_random_prompt(source_stories, prompt_len)
    batch_prompts = np.repeat(prompt_ids_1d[None, :], repeats=n_samples, axis=0).astype(np.int32)
    key           = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))

    rollout, _    = generate_rollout(params, jnp.asarray(batch_prompts, dtype=jnp.int32), key, 1.0)

    prompt_txt    = decode_ids(prompt_ids_1d)
    samples_txt   = [decode_ids(np.array(rollout[i], dtype=np.int32)) for i in range(n_samples)]
    return prompt_txt, samples_txt


def reward_hybrid(text, fluency_score):
    # Lexical Check
    t     = text.lower()
    words = re.findall(r"[a-z]+", t)

    r = 0.0
    for w in words:
        if w in happy_vocab: r += 2.0
        if w in sad_vocab:   r -= 1.0

    # Structure Bonus
    ex = t.count("!")
    r += 0.1 * min(ex, 3)
    if ex > 3: r -= 0.2 * float(ex - 3)

    if text.strip().endswith("."): r += 1.0

    # N-gram Loop Penalty
    if len(words) > 3:
        trigrams = set()
        for i in range(len(words) - 2):
            tg = (words[i], words[i+1], words[i+2])
            if tg in trigrams:
                return -5.0 # hard fail on loops
            trigrams.add(tg)

    # Fluency Gate
    if fluency_score < fluency_threshold:
        return -5.0 # reject garbage

    rr = float(r) + 0.5 * fluency_score
    rr = max(-reward_clip_max, min(reward_clip_max, rr))
    return float(rr)


def compute_masked_advantages(rewards, n_prompts, group_size, use_std_norm=True):
    rg   = rewards.reshape(n_prompts, group_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    std  = np.std (rg, axis=1, keepdims=True) + 1e-8

    adv  = (rg - mean) / std if use_std_norm else (rg - mean)

    # Mask: prevent cold-start stall by keeping slightly negative samples
    # if they are better than the group average.
    mask = (rg > -1.0).astype(np.float32)
    adv  = adv * mask

    adv  = np.clip(adv, -adv_clip_max, adv_clip_max)
    return adv.reshape(-1).astype(np.float32), float(mean.mean()), float(std.mean())


def grpo_all_story_prob(update):
    if update < mix_stage1_updates:
        return 0.30
    if update < mix_stage2_updates:
        return 0.50
    return 0.70


grpo_tx = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(grpo_lr)
)


@jax.jit
def grpo_train_step(state, frozen_ref_params, rollout_seqs, behavior_logps, advantages, kl_beta_curr, dropout_key):
    inputs    = rollout_seqs[:, :-1]
    targets   = rollout_seqs[:,  1:]
    gen_start = prompt_len - 1
    gen_end   = gen_start + gen_len

    def loss_fn(p):
        logits       = unroll_logits_train(p, inputs, dropout_key)
        logits_gen   = logits  [:, gen_start:gen_end, :]
        targ_gen     = targets [:, gen_start:gen_end]

        logp_all     = jax.nn.log_softmax(logits_gen, axis=-1)
        onehot       = jax.nn.one_hot(targ_gen, vocab_size)
        logp_new     = jnp.sum(logp_all * onehot, axis=-1)

        # Ratio calc (valid because grpo_temp=1.0)
        ratio        = jnp.exp(logp_new - behavior_logps)
        ratio_clip   = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        adv_sg       = jax.lax.stop_gradient(advantages)[:, None]

        surr1        = ratio      * adv_sg
        surr2        = ratio_clip * adv_sg
        pg_loss      = -jnp.mean(jnp.minimum(surr1, surr2))

        ent          = entropy_from_logits(logits_gen.reshape((-1, vocab_size)))

        # KL vs Frozen Ref
        ref_logits   = unroll_logits_eval(frozen_ref_params, inputs)[:, gen_start:gen_end, :]
        kl           = kl_from_logits(
            logits_gen.reshape((-1, vocab_size)),
            ref_logits.reshape((-1, vocab_size))
        )

        total_loss   = pg_loss + kl_beta_curr * kl - entropy_coeff * ent
        return total_loss, (pg_loss, kl, ent)

    (loss, (pg_loss, kl, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state                             = state.apply_gradients(grads=grads)
    return state, loss, pg_loss, kl, ent


def sample_inference(params, prompt_text, n_samples=3):
    prompt_ids = np.full((n_samples, prompt_len), pad_id, dtype=np.int32)
    enc        = encode_text(prompt_text)[:prompt_len]
    prompt_ids[:, :len(enc)] = np.array(enc, dtype=np.int32)
    prompt_ids[0] = ensure_valid_prompt(prompt_ids[0], min_prompt_tokens)
    for i in range(1, n_samples): prompt_ids[i] = prompt_ids[0]

    key        = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
    rollout, _ = generate_rollout(params, jnp.asarray(prompt_ids, dtype=jnp.int32), key, 1.0)
    return [decode_ids(np.array(rollout[i], dtype=np.int32)) for i in range(n_samples)]


print(f"Filtered (Positive) Stories: {len(positive_stories)} / {len(all_stories)}")
print(f"Vocab size                 : {vocab_size}")

print("\n=== PHASE 1: SFT (Bias to Positive Distribution) ===")
sft_drop_key = jax.random.PRNGKey(seed + 111)

for step in range(sft_total_steps):
    batch = sample_sft_batch(sft_batch_size, sft_seq_len)
    sft_drop_key, subk = jax.random.split(sft_drop_key)
    sft_state, sft_loss = sft_train_step(sft_state, jnp.asarray(batch, dtype=jnp.int32), subk)

    if debug and (step % 200 == 0):
        print(f"SFT {step:5d} | Loss {float(sft_loss):.4f}")

    if debug_render and (step % sft_print_freq == 0):
        ptxt, outs = sample_snapshot_from(sft_state.params, sft_print_count, positive_stories)
        print("SFT prompt :", ptxt[:160].replace("\n", " "))
        for i, s in enumerate(outs):
            print(f"SFT sample {i}:", s[:220].replace("\n", " "))


if debug_render:
    print("\nSFT sample (fixed prompt):")
    print(sample_inference(sft_state.params, "Once upon a time", n_samples=2)[0])


print("\n=== PHASE 2: GRPO (Critic-less PPO-clip + Masked Adv + Frozen Ref) ===")
grpo_state        = train_state.TrainState.create(apply_fn=model.apply, params=sft_state.params, tx=grpo_tx)
frozen_ref_params = grpo_state.params

grpo_drop_key     = jax.random.PRNGKey(seed + 222)

for update in range(grpo_total_updates):
    p_all = grpo_all_story_prob(update)

    # Mix prompts (positive bias vs general robustness)
    distinct_prompts_list = []
    for _ in range(prompts_per_update):
        src = all_stories if (np.random.rand() < p_all) else positive_stories
        distinct_prompts_list.append(sample_random_prompt(src, prompt_len))
    distinct_prompts_np   = np.stack(distinct_prompts_list)

    prompts_np = np.repeat(distinct_prompts_np, repeats=group_size, axis=0).astype(np.int32)
    prompts_j  = jnp.asarray(prompts_np, dtype=jnp.int32)

    actor_old_params = grpo_state.params

    key = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))

    # Rollout (temp=1.0)
    rollout_seqs, behavior_logps = generate_rollout(
        actor_old_params,
        prompts_j       ,
        key             ,
        temperature=grpo_temp
    )

    fluency_scores_j = compute_fluency_batch(frozen_ref_params, rollout_seqs)
    fluency_scores   = np.array(fluency_scores_j)

    rollout_ids      = np.array(rollout_seqs  , dtype=np.int32  )
    behavior_lp_np   = np.array(behavior_logps, dtype=np.float32)

    rewards          = np.zeros((rollout_ids.shape[0],), dtype=np.float32)
    for i in range(rollout_ids.shape[0]):
        # Hybrid Reward
        r = reward_hybrid(
            decode_ids(rollout_ids[i, prompt_len:]),
            float(fluency_scores[i])
        )
        rewards[i] = r

    # Masked Advantages
    advantages, mean_r, mean_std = compute_masked_advantages(
        rewards, prompts_per_update, group_size, use_std_advantage
    )

    full_j   = jnp.asarray(rollout_ids   , dtype=jnp.int32  )
    oldlp_j  = jnp.asarray(behavior_lp_np, dtype=jnp.float32)
    adv_j    = jnp.asarray(advantages    , dtype=jnp.float32)
    klb_j    = jnp.asarray(kl_beta       , dtype=jnp.float32)

    B_total  = int(full_j.shape[0])
    pg_loss = kl_loss = ent_loss = 0.0

    # PPO Update
    for _ in range(ppo_epochs):
        indices = np.random.permutation(B_total)
        for start in range(0, B_total, mini_batch_size):
            end = start + mini_batch_size
            if end > B_total: continue
            mb = indices[start:end]
            grpo_drop_key, subk = jax.random.split(grpo_drop_key)

            grpo_state, _, pg_loss, kl_loss, ent_loss = grpo_train_step(
                grpo_state       ,
                frozen_ref_params,
                full_j [mb]      ,
                oldlp_j[mb]      ,
                adv_j  [mb]      ,
                klb_j            ,
                subk
            )

    if debug and (update % 20 == 0):
        print(f"GRPO {update:4d} | R {mean_r:8.3f} | Std {mean_std:7.3f} | "
              f"PG {float(pg_loss):9.4f} | KL {float(kl_loss):9.4f} | Ent {float(ent_loss):9.4f}")

    if debug_render and (update % grpo_print_freq == 0):
        ptxt, outs = sample_snapshot_from(grpo_state.params, grpo_print_count, positive_stories)
        print("GRPO prompt (pos):", ptxt[:160].replace("\n", " "))
        for i, s in enumerate(outs):
            print(f"GRPO sample {i}:", s[:220].replace("\n", " "))


print("\n=== DONE ===")
final_samples = sample_inference(grpo_state.params, "Once upon a time", n_samples=3)
for i, s in enumerate(final_samples):
    print(f"Final {i}: {s[:260].replace(chr(10), ' ')}")