#
# TinyStories (char-level) + LSTM  ->  "Happy" Alignment via Critic-less PPO-GRPO (MVP)
#
# Goal:
#   Phase 1 (SFT)  : learn next-character distribution from tinystories-small.txt
#   Phase 2 (GRPO) : sample GROUP_SIZE completions per prompt, score with a simple "happy judge",
#                    compute outcome-based GRPO advantages, then PPO-clip update (NO critic)
#
# Notes:
#   - <|endoftext|> is used as separator in tinystories-small.txt
#   - Character-level tokens for simplicity
#   - PPO-style ratio uses stored OLD per-token log-probs from the behavior policy (actor_old_params)
#   - Advantage is outcome-based scalar per completion, broadcast across generated tokens
#

import os
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
eval_frequency        = 25

dataset_path          = "tinystories-small.txt"
end_of_text_token     = "<|endoftext|>"

seed                  = 42

# Phase 1: SFT
sft_updates           = 1800
sft_batch_size        = 64
sft_seq_len           = 128
sft_learning_rate     = 0.002

# Phase 2: GRPO + PPO-clip (criticless)
grpo_updates          = 200
prompts_per_update    = 16
group_size            = 8

prompt_len            = 32
gen_len               = 64

ppo_epochs            = 4
mini_batch_size       = 128
clip_epsilon          = 0.2

entropy_coefficient   = 0.01
kl_beta               = 0.02
ref_update_freq       = 10

grpo_learning_rate    = 0.0005
max_grad_norm         = 0.5
use_std_advantage     = True

# Reward definitions (very simple "judge")
happy_words           = [
    "happy","joy","joyful","smile","smiled","laugh","laughed","love","loved","kind","nice","fun",
    "good","great","amazing","wonderful","excited","brave","bright","safe","friend","friends"
]
sad_words             = [
    "sad","cry","cried","bad","angry","mad","hurt","scary","afraid","fear","dark",
    "hate","hated","mean","alone","lost","dead","death","kill","killed"
]


np.random.seed(seed)
random.seed(seed)


if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

stories = [s.strip() for s in raw_text.split(end_of_text_token)]
stories = [s for s in stories if len(s) > 0]

all_chars = set()
for s in stories:
    for ch in s:
        all_chars.add(ch)

PAD = "<PAD>"
BOS = "<BOS>"

chars = [PAD, BOS] + sorted(list(all_chars))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}

vocab_size  = len(chars)
pad_id      = char_to_id[PAD]
bos_id      = char_to_id[BOS]


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


corpus       = "\n\n".join(stories)
corpus_ids   = np.array(encode_text(corpus), dtype=np.int32)


embed_dim    = 32
hidden_dim   = 128


class TinyLSTM(nn.Module):
    vocab_size : int
    embed_dim  : int
    hidden_dim : int

    @nn.compact
    def __call__(self, token_id, carry):
        x              = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(token_id)
        carry, h        = nn.LSTMCell(features=self.hidden_dim)(carry, x)
        logits          = nn.Dense(features=self.vocab_size)(h)
        return logits, carry


def init_carry(batch_size):
    # Important: carry must be batched (B, H), otherwise lax.scan will error.
    c = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    h = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    return (c, h)


model        = TinyLSTM(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)

dummy_token  = jnp.zeros((1,), dtype=jnp.int32)
dummy_carry  = init_carry(1)
params       = model.init(jax.random.PRNGKey(seed), dummy_token, dummy_carry)["params"]


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
def unroll_logits(params, token_seq):
    # token_seq: (B, T) token ids
    B, T  = token_seq.shape
    carry = init_carry(B)

    def step_fn(carry, tok):
        logits, carry = model.apply({"params": params}, tok, carry)
        return carry, logits

    carry, logits_seq = jax.lax.scan(step_fn, carry, token_seq.T)              # (T, B, V)
    logits_seq        = jnp.transpose(logits_seq, (1, 0, 2))                   # (B, T, V)
    return logits_seq


sft_tx       = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(sft_learning_rate)
)
sft_state    = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sft_tx)


def sample_sft_batch(batch_size, seq_len):
    # Need (seq_len + 1) tokens so we can do next-token targets.
    need     = int(seq_len + 1)
    N        = int(corpus_ids.shape[0])

    if N <= need + 1:
        # safety for tiny corpus
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
def sft_train_step(state, batch_tokens):
    inputs  = batch_tokens[:, :-1]
    targets = batch_tokens[:,  1:]

    def loss_fn(p):
        logits = unroll_logits(p, inputs)                                      # (B, T, V)
        logp   = jax.nn.log_softmax(logits, axis=-1)
        onehot = jax.nn.one_hot(targets, vocab_size)
        nll    = -jnp.sum(logp * onehot, axis=-1)                              # (B, T)

        # Here PAD is rare in SFT batches, but we keep a mask for correctness.
        mask   = (targets != pad_id).astype(jnp.float32)
        return jnp.sum(nll * mask) / (jnp.sum(mask) + 1e-8)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state       = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def condition_on_prompt(params, prompt_tokens):
    # prompt_tokens: (B, P)
    B, P  = prompt_tokens.shape
    carry = init_carry(B)

    def step_fn(carry, tok):
        logits, carry = model.apply({"params": params}, tok, carry)
        return carry, logits

    carry, _ = jax.lax.scan(step_fn, carry, prompt_tokens.T)
    return carry


@jax.jit
def generate_from_prompts(actor_old_params, prompt_tokens, key):
    # prompt_tokens: (N, P)
    N, P     = prompt_tokens.shape
    carry    = condition_on_prompt(actor_old_params, prompt_tokens)
    last_tok = prompt_tokens[:, -1]

    def gen_step(carry_key_tok, _):
        carry, key, tok = carry_key_tok
        logits, carry   = model.apply({"params": actor_old_params}, tok, carry)
        key, subkey     = jax.random.split(key)
        next_tok        = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        lp              = logprob_from_logits(logits, next_tok)                # (N,)
        return (carry, key, next_tok), (next_tok, lp)

    (carry, key, _), (gen_toks, gen_lps) = jax.lax.scan(
        gen_step, (carry, key, last_tok), xs=None, length=gen_len
    )

    gen_toks    = jnp.transpose(gen_toks, (1, 0))                              # (N, gen_len)
    gen_lps     = jnp.transpose(gen_lps , (1, 0))                              # (N, gen_len)
    full_tokens = jnp.concatenate([prompt_tokens, gen_toks], axis=1)           # (N, P+gen_len)
    return full_tokens, gen_lps


def sample_prompts(n_prompts, prompt_len):
    prompts = np.full((n_prompts, prompt_len), pad_id, dtype=np.int32)

    for i in range(n_prompts):
        s   = stories[np.random.randint(0, len(stories))].replace("\r", "")
        ids = encode_text(s)

        if len(ids) <= 1:
            prompts[i, 0] = bos_id
            continue

        if len(ids) < (prompt_len + 1):
            take = ids[:prompt_len]
        else:
            start_max = max(1, len(ids) - prompt_len)
            start     = np.random.randint(0, start_max)
            take      = ids[start:start + prompt_len]

        take = take[:prompt_len]
        prompts[i, :len(take)] = np.array(take, dtype=np.int32)

        if prompts[i, 0] != bos_id:
            prompts[i, 0] = bos_id

    return prompts


def reward_happy(text):
    # Very simple "judge": count whole-word matches.
    t      = text.lower()
    words  = re.findall(r"[a-z]+", t)

    r      = 0.0
    for w in words:
        if w in happy_words: r += 2.0
        if w in sad_words  : r -= 1.0

    r += 0.1 * t.count("!")
    return float(r)


def compute_group_advantages(rewards, prompts_per_update, group_size, use_std_advantage=True):
    rg   = rewards.reshape(prompts_per_update, group_size)
    mean = np.mean(rg, axis=1, keepdims=True)
    std  = np.std (rg, axis=1, keepdims=True) + 1e-8

    adv  = (rg - mean) / std if use_std_advantage else (rg - mean)

    return adv.reshape(-1).astype(np.float32), float(mean.mean()), float(std.mean())


grpo_tx       = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(grpo_learning_rate)
)


@jax.jit
def grpo_train_step(state, actor_ref_params, batch_full_tokens, batch_old_logps, batch_advantages, kl_beta_dynamic):
    # batch_full_tokens : (B, prompt_len + gen_len)
    # batch_old_logps   : (B, gen_len)  logp under behavior policy for the generated tokens
    # batch_advantages  : (B,)          scalar outcome advantage per completion

    inputs      = batch_full_tokens[:, :-1]
    targets     = batch_full_tokens[:,  1:]

    gen_start   = prompt_len - 1
    gen_end     = gen_start + gen_len

    def loss_fn(p):
        logits       = unroll_logits(p, inputs)                                # (B, T, V)
        logits_gen   = logits  [:, gen_start:gen_end, :]
        targ_gen     = targets [:, gen_start:gen_end]                          # (B, gen_len)

        logp_all     = jax.nn.log_softmax(logits_gen, axis=-1)
        onehot       = jax.nn.one_hot(targ_gen, vocab_size)
        logp_new     = jnp.sum(logp_all * onehot, axis=-1)                     # (B, gen_len)

        ratio        = jnp.exp(logp_new - batch_old_logps)
        ratio_clip   = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        adv_sg       = jax.lax.stop_gradient(batch_advantages)[:, None]        # (B, 1)

        surr1        = ratio      * adv_sg
        surr2        = ratio_clip * adv_sg
        pg_loss      = -jnp.mean(jnp.minimum(surr1, surr2))

        ent          = entropy_from_logits(logits_gen.reshape((-1, vocab_size)))

        ref_logits   = unroll_logits(actor_ref_params, inputs)[:, gen_start:gen_end, :]
        kl           = kl_from_logits(
            logits_gen.reshape((-1, vocab_size)),
            ref_logits.reshape((-1, vocab_size))
        )

        total_loss   = pg_loss + kl_beta_dynamic * kl - entropy_coefficient * ent
        return total_loss, (pg_loss, kl, ent)

    (loss, (pg_loss, kl, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state                             = state.apply_gradients(grads=grads)
    return state, loss, pg_loss, kl, ent


def sample_one(params, prompt_text, n_samples=3):
    prompt_ids = np.full((n_samples, prompt_len), pad_id, dtype=np.int32)
    enc        = encode_text(prompt_text)[:prompt_len]
    prompt_ids[:, :len(enc)] = np.array(enc, dtype=np.int32)

    key        = jax.random.PRNGKey(np.random.randint(0, 100000))
    full, _    = generate_from_prompts(params, jnp.asarray(prompt_ids, dtype=jnp.int32), key)
    return [decode_ids(np.array(full[i], dtype=np.int32)) for i in range(n_samples)]


print(f"Loaded stories: {len(stories)}")
print(f"Vocab size   : {vocab_size}")

print("\n=== PHASE 1: SFT (learn TinyStories character distribution) ===")
for step in range(sft_updates):
    batch               = sample_sft_batch(sft_batch_size, sft_seq_len)
    sft_state, sft_loss = sft_train_step(sft_state, jnp.asarray(batch, dtype=jnp.int32))

    if debug and (step % 200 == 0):
        print(f"SFT {step:5d} | Loss {float(sft_loss):.4f}")

if debug_render:
    print("\nSFT sample:")
    print(sample_one(sft_state.params, "Once upon a time", n_samples=2)[0])


print("\n=== PHASE 2: GRPO (Critic-less PPO-clip) ===")
grpo_state       = train_state.TrainState.create(apply_fn=model.apply, params=sft_state.params, tx=grpo_tx)
actor_ref_params = grpo_state.params

for update in range(grpo_updates):

    if (kl_beta > 0.0) and (update % ref_update_freq == 0):
        actor_ref_params = grpo_state.params

    base_prompts = sample_prompts(prompts_per_update, prompt_len)
    prompts      = np.repeat(base_prompts, repeats=group_size, axis=0).astype(np.int32)

    # Behavior policy snapshot for PPO ratios
    actor_old_params = grpo_state.params

    key               = jax.random.PRNGKey(np.random.randint(0, 100000))
    full_tokens, old_logps = generate_from_prompts(
        actor_old_params,
        jnp.asarray(prompts, dtype=jnp.int32),
        key
    )

    full_np           = np.array(full_tokens, dtype=np.int32)
    oldlp_np          = np.array(old_logps  , dtype=np.float32)

    rewards           = np.zeros((full_np.shape[0],), dtype=np.float32)
    for i in range(full_np.shape[0]):
        rewards[i] = float(reward_happy(decode_ids(full_np[i])))

    advantages, mean_r, mean_std = compute_group_advantages(
        rewards, prompts_per_update, group_size, use_std_advantage
    )

    full_j            = jnp.asarray(full_np, dtype=jnp.int32)
    oldlp_j           = jnp.asarray(oldlp_np, dtype=jnp.float32)
    adv_j             = jnp.asarray(advantages, dtype=jnp.float32)
    klb_j             = jnp.asarray(kl_beta, dtype=jnp.float32)

    B_total           = int(full_j.shape[0])

    # PPO-style multiple epochs over the same sampled batch
    pg_loss           = 0.0
    kl_loss           = 0.0
    ent_loss          = 0.0

    for _ in range(ppo_epochs):
        indices = np.random.permutation(B_total)

        for start in range(0, B_total, mini_batch_size):
            end = start + mini_batch_size
            if end > B_total:
                continue

            mb = indices[start:end]

            grpo_state, _, pg_loss, kl_loss, ent_loss = grpo_train_step(
                grpo_state      ,
                actor_ref_params,
                full_j [mb]     ,
                oldlp_j[mb]     ,
                adv_j  [mb]     ,
                klb_j
            )

    if debug and (update % 20 == 0):
        print(f"GRPO {update:4d} | R {mean_r:8.3f} | Std {mean_std:7.3f} | "
              f"PG {float(pg_loss):9.4f} | KL {float(kl_loss):9.4f} | Ent {float(ent_loss):9.4f}")

    if debug_render and (update % eval_frequency == 0):
        samples = sample_one(grpo_state.params, "One day", n_samples=2)
        print("Sample 1:", samples[0][:200].replace("\n", " "))
        print("Sample 2:", samples[1][:200].replace("\n", " "))


print("\n=== DONE ===")
final_samples = sample_one(grpo_state.params, "Once upon a time", n_samples=3)
for i, s in enumerate(final_samples):
    print(f"Final {i}: {s[:250].replace(chr(10), ' ')}")
