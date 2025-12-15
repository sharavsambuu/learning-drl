#
# Rainbow DQN 
#
# Components: 
#   1) Double DQN (Online selects, Target evaluates)
#   2) Prioritized Experience Replay (PER) (SumTree)
#   3) Dueling Network (Value + Advantage)
#   4) Noisy Networks (Factorized Gaussian NoisyNet, Fortunato-style)
#   5) C51 Distributional RL (Categorical value distribution)
#   6) N-step Returns (stores per-sample gamma_n = gamma^n_used)
#
# Notes:
#   - CartPole rewards are +1 per step. A good C51 support is [0, 500].
#   - Truncation (time-limit) is not "death" for learning in general;
#     but for CartPole, it usually means success. We treat:
#       done_boundary = terminated OR truncated   (reset env)
#       done_term     = terminated                (mask bootstrap)
#
#

import random
import time
from collections import deque
import jax
import optax
import numpy             as np
import flax.linen        as nn
from   jax               import numpy as jnp
from   flax.linen        import initializers
import gymnasium         as gym
import matplotlib.pyplot as plt


# Hyperparameters 
debug_render        = True
live_plot           = True
plot_every_steps    = 50

num_episodes        = 500
batch_size          = 64
learning_rate       = 0.00025

sync_steps          = 2000
train_every_steps   = 4
memory_length       = 20000

n_steps             = 3
gamma               = 0.99

# C51 / Distributional
v_min               = 0.0
v_max               = 500.0
n_atoms             = 51

z_holder            = jnp.linspace(v_min, v_max, n_atoms, dtype=jnp.float32)
dz                  = (v_max - v_min) / (n_atoms - 1)

# PER
per_e               = 0.01
per_a               = 0.6
per_b_start         = 0.4
per_b_end           = 1.0
per_b_decay         = num_episodes

# Noisy Nets
sigma_init          = 0.5

# Optimizer / Stability
max_grad_norm       = 10.0


class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data     = np.empty(capacity, dtype=object)
        self.data[:]  = None
        self.size     = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return float(self.tree[0])

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx     = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, float(self.tree[idx]), self.data[dataIdx])

class PERMemory:
    e    = per_e
    a    = per_a
    beta = per_b_start

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n, episode):
        if self.tree.size < n:
            return [], [], np.array([], dtype=np.float32)

        frac     = min(episode / per_b_decay, 1.0)
        self.beta = per_b_start + frac * (per_b_end - per_b_start)

        batch      = []
        idxs       = []
        is_weights = []

        total_p    = self.tree.total()
        segment    = total_p / n

        leaves = self.tree.tree[-self.tree.capacity:]
        valid  = leaves[leaves > 0]
        p_min  = (np.min(valid) / total_p) if valid.size > 0 else (1.0 / total_p)
        max_w  = (p_min * self.tree.size) ** (-self.beta)

        # Retry loop so we reliably return n samples even with partially-empty buffer early on
        tries = 0
        while len(batch) < n and tries < n * 5:
            i = len(batch)
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            if data is None:
                tries += 1
                continue

            prob   = p / total_p
            w      = (prob * self.tree.size) ** (-self.beta)

            batch.append(data)
            idxs.append(idx)
            is_weights.append(w / max_w)

        return batch, idxs, np.array(is_weights, dtype=np.float32)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


# Networks (Factorized NoisyNet + Dueling + C51 logits)

def f_noise(x):
    # Fortunato factorized noise transform: f(x) = sign(x) * sqrt(|x|)
    return jnp.sign(x) * jnp.sqrt(jnp.abs(x) + 1e-10)

def uniform(scale=0.05):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    return init

class NoisyDense(nn.Module):
    features:   int
    sigma_init: float = sigma_init

    @nn.compact
    def __call__(self, x, noise_key):
        in_features  = x.shape[-1]
        sigma_scale  = self.sigma_init / np.sqrt(in_features)

        w_mu         = self.param("w_mu"   , initializers.lecun_uniform(), (in_features, self.features))
        b_mu         = self.param("b_mu"   , initializers.zeros          , (self.features,))

        w_sigma      = self.param("w_sigma", uniform(sigma_scale)        , (in_features, self.features))
        b_sigma      = self.param("b_sigma", uniform(sigma_scale)        , (self.features,))

        k1, k2       = jax.random.split(noise_key)

        eps_in       = f_noise(jax.random.normal(k1, (in_features,), dtype=x.dtype))
        eps_out      = f_noise(jax.random.normal(k2, (self.features,), dtype=x.dtype))

        w_eps        = jnp.outer(eps_in, eps_out)
        b_eps        = eps_out

        w            = w_mu + w_sigma * w_eps
        b            = b_mu + b_sigma * b_eps

        return jnp.dot(x, w) + b


class NoisyDuelingQNetwork(nn.Module):
    n_actions: int
    n_atoms:   int

    @nn.compact
    def __call__(self, x, noise_key):
        nk1, nk2, nk3, nk4, nk5, nk6 = jax.random.split(noise_key, 6)

        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = NoisyDense(64)(x, nk1)
        x = nn.relu(x)

        # Value stream: (B, 1, Atoms)
        v = NoisyDense(64)(x, nk2)
        v = nn.relu(v)
        v = NoisyDense(self.n_atoms)(v, nk3)
        v = v.reshape((-1, 1, self.n_atoms))

        # Advantage stream: (B, Actions, Atoms)
        a = NoisyDense(64)(x, nk4)
        a = nn.relu(a)
        a = NoisyDense(self.n_actions * self.n_atoms)(a, nk5)
        a = a.reshape((-1, self.n_actions, self.n_atoms))

        q = v + (a - jnp.mean(a, axis=1, keepdims=True))

        # Output is logits over atoms (C51)
        return q



env        = gym.make("CartPole-v1", render_mode="human" if debug_render else None)
n_actions  = env.action_space.n
state_dim  = env.observation_space.shape[0]

dqn_module = NoisyDuelingQNetwork(n_actions=n_actions, n_atoms=n_atoms)

dummy_x    = jnp.zeros((1, state_dim), dtype=jnp.float32)

rng        = jax.random.PRNGKey(0)
rng, i_rng, n_rng = jax.random.split(rng, 3)
params     = dqn_module.init({"params": i_rng}, dummy_x, noise_key=n_rng)["params"]

q_params   = params
t_params   = params

optimizer  = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(learning_rate, eps=1e-5)
)
opt_state  = optimizer.init(q_params)


@jax.jit
def get_action_and_probs(params, x, noise_key):
    logits = dqn_module.apply({"params": params}, x, noise_key=noise_key)     # (1, A, Atoms)
    probs  = jax.nn.softmax(logits, axis=-1)                                  # (1, A, Atoms)
    q_vals = jnp.sum(probs * z_holder, axis=-1)                               # (1, A)
    act    = jnp.argmax(q_vals, axis=1)[0]
    return act, probs

def project_sample(next_probs_atom, reward, done, gamma_n):
    # Canonical C51 projection for one sample (vectorized over atoms)
    Tz = reward + (1.0 - done) * gamma_n * z_holder
    Tz = jnp.clip(Tz, v_min, v_max)

    b  = (Tz - v_min) / dz
    l  = jnp.floor(b).astype(jnp.int32)
    u  = jnp.ceil (b).astype(jnp.int32)

    l  = jnp.clip(l, 0, n_atoms - 1)
    u  = jnp.clip(u, 0, n_atoms - 1)

    m_u = (b - l.astype(jnp.float32))
    m_l = (u.astype(jnp.float32) - b)

    same = (u == l)
    m_l  = jnp.where(same, 1.0, m_l)
    m_u  = jnp.where(same, 0.0, m_u)

    proj = jnp.zeros((n_atoms,), dtype=jnp.float32)
    proj = proj.at[l].add(next_probs_atom * m_l)
    proj = proj.at[u].add(next_probs_atom * m_u)

    return proj

@jax.jit
def train_step(q_params, target_params, opt_state, batch, is_weights, noise_key):
    states, actions, rewards, next_states, dones, gamma_ns = batch

    nk_curr, nk_online, nk_target = jax.random.split(noise_key, 3)

    # Double DQN
    on_logits = dqn_module.apply({"params": q_params}, next_states, noise_key=nk_online)
    on_probs  = jax.nn.softmax(on_logits, axis=-1)
    on_q      = jnp.sum(on_probs * z_holder, axis=-1)
    best_acts = jnp.argmax(on_q, axis=1)

    tg_logits = dqn_module.apply({"params": target_params}, next_states, noise_key=nk_target)
    tg_probs  = jax.nn.softmax(tg_logits, axis=-1)

    best_ix   = best_acts[:, None, None]
    best_ix   = jnp.broadcast_to(best_ix, (best_acts.shape[0], 1, n_atoms))
    tg_best   = jnp.take_along_axis(tg_probs, best_ix, axis=1).squeeze(1)     # (B, Atoms)

    # C51 Projection (per-sample gamma^n)
    targets   = jax.vmap(project_sample, in_axes=(0, 0, 0, 0))(
        tg_best, rewards, dones, gamma_ns
    )

    def loss_fn(params, key):
        curr_logits = dqn_module.apply({"params": params}, states, noise_key=key)
        curr_logp   = jax.nn.log_softmax(curr_logits, axis=-1)                 # (B, A, Atoms)

        act_ix      = actions[:, None, None]
        act_ix      = jnp.broadcast_to(act_ix, (actions.shape[0], 1, n_atoms))
        curr_logp_a = jnp.take_along_axis(curr_logp, act_ix, axis=1).squeeze(1)

        element_loss = -jnp.sum(targets * curr_logp_a, axis=1)                 # (B,)
        loss         = jnp.mean(element_loss * is_weights)

        return loss, element_loss

    (loss, per_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_params, nk_curr)

    updates, opt_state = optimizer.update(grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)

    return q_params, opt_state, loss, per_loss



if live_plot:
    plt.ion()
    fig, axes = plt.subplots(1, n_actions, figsize=(10, 4), sharey=True)

    if n_actions == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.set_xlim(float(v_min), float(v_max))
        ax.set_ylim(0.0, 0.35)
        ax.set_xlabel("Value (Z)")
        ax.set_ylabel("Prob")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Action {i}")

    z_np  = np.array(z_holder)
    bars  = []
    for ax in axes:
        bs = ax.bar(z_np, np.zeros_like(z_np), width=dz, alpha=0.7)
        bars.append(bs)

    fig.tight_layout()
    plt.show(block=False)



per_memory    = PERMemory(memory_length)
n_step_memory = deque(maxlen=n_steps)

global_steps  = 0
ep_rewards    = deque(maxlen=100)

try:
    for episode in range(num_episodes):
        state, _  = env.reset()
        state     = np.array(state, dtype=np.float32)
        ep_ret    = 0.0

        while True:
            global_steps += 1

            rng, noise_key = jax.random.split(rng)
            action, probs  = get_action_and_probs(q_params, state[None, :], noise_key)
            action         = int(action)

            if live_plot and (global_steps % plot_every_steps == 0):
                p = np.array(probs[0])                                          # (A, Atoms)
                for a in range(n_actions):
                    for bar, h in zip(bars[a], p[a]):
                        bar.set_height(h)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done_boundary = bool(terminated or truncated)                       # reset condition
            done_term     = bool(terminated)                                    # learning mask

            next_state = np.array(next_state, dtype=np.float32)

            # Store raw step for N-step rolling buffer
            n_step_memory.append((state, action, float(reward), next_state, float(done_term)))

            # N-step processing:
            #   - For normal flow: emit exactly 1 transition when buffer is full
            #   - For boundary: flush everything remaining (with correct gamma^n_used)
            if (len(n_step_memory) == n_steps) or done_boundary:
                while len(n_step_memory) > 0:
                    s0, a0, _, _, _ = n_step_memory[0]

                    R        = 0.0
                    n_used   = len(n_step_memory)

                    for i, exp in enumerate(n_step_memory):
                        R += exp[2] * (gamma ** i)

                    sn       = n_step_memory[-1][3]
                    dn       = n_step_memory[-1][4]

                    gamma_n  = float(gamma ** n_used)

                    # Store: (s, a, nstep_return, s_n, done_term, gamma^n_used)
                    per_memory.add(1.0, (s0, a0, R, sn, dn, gamma_n))

                    n_step_memory.popleft()

                    if not done_boundary:
                        break

            state   = next_state
            ep_ret += float(reward)

            # Train
            if per_memory.tree.size >= batch_size and (global_steps % train_every_steps == 0):
                batch, idxs, weights = per_memory.sample(batch_size, episode)
                if len(batch) == batch_size:
                    b_s, b_a, b_r, b_ns, b_d, b_gn = zip(*batch)

                    b_s   = jnp.asarray(b_s , dtype=jnp.float32)
                    b_a   = jnp.asarray(b_a , dtype=jnp.int32  )
                    b_r   = jnp.asarray(b_r , dtype=jnp.float32)
                    b_ns  = jnp.asarray(b_ns, dtype=jnp.float32)
                    b_d   = jnp.asarray(b_d , dtype=jnp.float32)
                    b_gn  = jnp.asarray(b_gn, dtype=jnp.float32)

                    w_j   = jnp.asarray(weights, dtype=jnp.float32)

                    rng, nk_train = jax.random.split(rng)
                    q_params, opt_state, loss, per_loss = train_step(
                        q_params, t_params, opt_state,
                        (b_s, b_a, b_r, b_ns, b_d, b_gn),
                        w_j, nk_train
                    )

                    # Update PER priorities from per-sample cross-entropy (C51 standard)
                    per_err = np.array(per_loss, dtype=np.float32) + 1e-6
                    for i in range(batch_size):
                        per_memory.update(idxs[i], per_err[i])

            # Target sync
            if global_steps % sync_steps == 0:
                t_params = q_params
                print(f"Step {global_steps}: Synced Target")

            if debug_render:
                env.render()

            if done_boundary:
                ep_rewards.append(ep_ret)
                if episode % 10 == 0:
                    print(f"Ep {episode:4d} | Avg: {np.mean(ep_rewards):6.1f} | Beta: {per_memory.beta:.2f} | Steps: {global_steps}")
                break

finally:
    env.close()
    if live_plot:
        plt.close()
