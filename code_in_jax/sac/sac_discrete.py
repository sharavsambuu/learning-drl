#
# SAC (Discrete) + Entropy Plot
#
#    - Discrete SAC target uses exact expectation over actions (no action sampling in target)
#    - Twin Q networks + clipped double Q
#    - Auto-tuned alpha (entropy temperature) with log_alpha clipping safety
#    - Prioritized Experience Replay (PER) SumTree (new samples start at max priority)
#    - Correct boundary handling:
#         terminated  -> done_term  (mask bootstrap)
#         truncated   -> boundary reset only (bootstrap allowed)
#    - Gradient clipping + Adam(eps=1e-5)
#    - Live plotting of entropy vs target entropy (and alpha on a twin axis)
#

import random
import time
import jax
import optax
import numpy             as np
import gymnasium         as gym
import flax.linen        as nn
from   jax               import numpy as jnp
import matplotlib.pyplot as plt


# Hyperparameters
env_name            = "LunarLander-v3"
debug_render        = True

live_plot           = True
plot_every_episodes = 1

total_timesteps     = 600000
start_steps         = 5000

learning_rate       = 3e-4
gamma               = 0.99
tau                 = 0.005

batch_size          = 64
memory_length       = 100000
update_every_steps  = 1
updates_per_step    = 1

# PER
per_alpha           = 0.6
per_beta_start      = 0.4
per_beta_frames     = 100000
per_epsilon         = 1e-6

# Entropy / Alpha
ALPHA_AUTO          = True
target_entropy_ratio= 0.98
init_alpha          = 0.2
alpha_lr            = 3e-4

max_grad_norm       = 10.0



class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Initialize with None to safely detect empty slots
        self.data     = np.full(capacity, None, dtype=object)
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


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=per_alpha):
        self.tree   = SumTree(capacity)
        self.alpha  = alpha
        self.max_p  = 1.0

    def _get_priority(self, error):
        return (np.abs(error) + per_epsilon) ** self.alpha

    def push(self, state, action, reward, next_state, done_term):
        p = self.max_p
        self.tree.add(p, (state, action, reward, next_state, done_term))

    def sample(self, batch_size, beta):
        if self.tree.size < batch_size:
            return None

        batch      = []
        idxs       = []
        priorities = []
        total_p    = self.tree.total()
        
        # Avoid division by zero
        if total_p <= 0.0:
            return None

        segment    = total_p / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            # Safety check for uninitialized data
            if data is None:
                return None

            batch     .append(data)
            idxs      .append(idx)
            priorities.append(p)

        probs   = np.array(priorities, dtype=np.float32) / total_p
        weights = (self.tree.size * probs) ** (-beta)
        weights = weights / weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            idxs,
            np.array(states     , dtype=np.float32),
            np.array(actions    , dtype=np.int32  ),
            np.array(rewards    , dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones      , dtype=np.float32),
            np.array(weights    , dtype=np.float32)
        )

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        if p > self.max_p:
            self.max_p = float(p)

    def __len__(self):
        return self.tree.size



class SoftQNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x

class DiscretePolicyNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)   # logits
        return x



env         = gym.make(env_name, render_mode="human" if debug_render else None)
state_dim   = env.observation_space.shape[0]
n_actions   = env.action_space.n

q1_module   = SoftQNetwork(n_actions)
q2_module   = SoftQNetwork(n_actions)
pi_module   = DiscretePolicyNetwork(n_actions)

rng         = jax.random.PRNGKey(42)
rng, k1, k2, k3 = jax.random.split(rng, 4)

dummy       = jnp.zeros((1, state_dim), dtype=jnp.float32)

q1_params   = q1_module.init(k1, dummy)["params"]
q2_params   = q2_module.init(k2, dummy)["params"]
tq1_params  = q1_params
tq2_params  = q2_params
pi_params   = pi_module.init(k3, dummy)["params"]

target_entropy = float(np.log(n_actions) * target_entropy_ratio)

if ALPHA_AUTO:
    log_alpha    = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    alpha_optim  = optax.adam(alpha_lr, eps=1e-5)
    alpha_opt_st = alpha_optim.init(log_alpha)
else:
    log_alpha    = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    alpha_optim  = None
    alpha_opt_st = None

optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(learning_rate, eps=1e-5)
)

q1_opt_st  = optimizer.init(q1_params)
q2_opt_st  = optimizer.init(q2_params)
pi_opt_st  = optimizer.init(pi_params)



@jax.jit
def soft_update(target, source, tau):
    return jax.tree_util.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)

@jax.jit
def sample_action(rng, pi_params, state):
    logits = pi_module.apply({"params": pi_params}, state)
    act    = jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)
    return act

@jax.jit
def update_step(
    q1_params , q1_opt_st  ,
    q2_params , q2_opt_st  ,
    pi_params , pi_opt_st  ,
    tq1_params, tq2_params ,
    log_alpha , alpha_opt_st,
    states, actions, rewards, next_states, dones, is_weights
):
    alpha = jnp.exp(log_alpha)

    # Target V(s') = sum_a pi(a|s') [ minQ(s',a) - alpha * log_pi(a|s') ]
    next_logits = pi_module.apply({"params": pi_params}, next_states)
    next_logp   = jax.nn.log_softmax(next_logits, axis=-1)
    next_probs  = jnp.exp(next_logp)

    tq1_next    = q1_module.apply({"params": tq1_params}, next_states)
    tq2_next    = q2_module.apply({"params": tq2_params}, next_states)
    tq_min      = jnp.minimum(tq1_next, tq2_next)

    next_v      = jnp.sum(next_probs * (tq_min - alpha * next_logp), axis=1)
    target_q    = rewards + gamma * (1.0 - dones) * next_v

    def q_loss_fn(p, apply_fn):
        q_pred = apply_fn({"params": p}, states)
        q_act  = jnp.take_along_axis(q_pred, actions[:, None], axis=1).squeeze(1)
        td     = target_q - q_act
        loss   = jnp.mean(is_weights * jnp.square(td))
        return loss, td

    (q1_loss, td1), g1 = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, q1_module.apply)
    u1, q1_opt_st      = optimizer.update(g1, q1_opt_st, q1_params)
    q1_params          = optax.apply_updates(q1_params, u1)

    (q2_loss, td2), g2 = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, q2_module.apply)
    u2, q2_opt_st      = optimizer.update(g2, q2_opt_st, q2_params)
    q2_params          = optax.apply_updates(q2_params, u2)

    per_err = 0.5 * (jnp.abs(td1) + jnp.abs(td2))

    def pi_loss_fn(p):
        logits      = pi_module.apply({"params": p}, states)
        logp        = jax.nn.log_softmax(logits, axis=-1)
        probs       = jnp.exp(logp)

        q1_curr     = q1_module.apply({"params": q1_params}, states)
        q2_curr     = q2_module.apply({"params": q2_params}, states)
        q_min       = jnp.minimum(q1_curr, q2_curr)

        policy_loss = jnp.sum(probs * (alpha * logp - q_min), axis=1)
        entropy     = -jnp.sum(probs * logp, axis=1)

        return jnp.mean(policy_loss), jnp.mean(entropy)

    (pi_loss, ent), g_pi = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
    u_pi, pi_opt_st      = optimizer.update(g_pi, pi_opt_st, pi_params)
    pi_params            = optax.apply_updates(pi_params, u_pi)

    alpha_loss = jnp.array(0.0, dtype=jnp.float32)
    if ALPHA_AUTO:
        def alpha_loss_fn(log_a):
            return jnp.exp(log_a) * (jax.lax.stop_gradient(ent) - target_entropy)

        alpha_loss, g_a   = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        u_a, alpha_opt_st = alpha_optim.update(g_a, alpha_opt_st, log_alpha)
        log_alpha         = optax.apply_updates(log_alpha, u_a)

        log_alpha = jnp.clip(log_alpha, -20.0, 2.0)

    tq1_params = soft_update(tq1_params, q1_params, tau)
    tq2_params = soft_update(tq2_params, q2_params, tau)

    return (
        q1_params, q1_opt_st   ,
        q2_params, q2_opt_st   ,
        pi_params, pi_opt_st   ,
        tq1_params, tq2_params ,
        log_alpha, alpha_opt_st,
        q1_loss, q2_loss, pi_loss, ent, per_err, alpha_loss
    )



if live_plot:
    plt.ion()
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

    ax1.set_title("SAC Discrete: Entropy vs Target (and Alpha)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Entropy")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Alpha")

    ent_line, = ax1.plot([], [], linewidth=2)
    tgt_line, = ax1.plot([], [], linestyle="--", linewidth=1)

    alp_line, = ax2.plot([], [], linewidth=1)

    fig.tight_layout()
    plt.show(block=False)

plot_eps   = []
plot_ent   = []
plot_alpha = []


replay      = PrioritizedReplayMemory(memory_length)

global_step = 0
episode     = 0

ep_ret      = 0.0
ep_ents     = []

try:
    obs, _ = env.reset()
    obs    = np.array(obs, dtype=np.float32)

    last_l_q1   = 0.0
    last_l_q2   = 0.0
    last_l_pi   = 0.0
    last_ent    = 0.0
    last_a_loss = 0.0

    while global_step < total_timesteps:
        global_step += 1

        if global_step < start_steps:
            action = env.action_space.sample()
        else:
            rng, a_rng = jax.random.split(rng)
            action = int(sample_action(a_rng, pi_params, obs[None, :])[0])

        next_obs, reward, terminated, truncated, _ = env.step(action)

        done_boundary = bool(terminated or truncated)
        done_term     = bool(terminated)

        next_obs = np.array(next_obs, dtype=np.float32)

        replay.push(obs, action, float(reward), next_obs, float(done_term))

        obs     = next_obs
        ep_ret += float(reward)

        if debug_render:
            env.render()

        if len(replay) >= batch_size and global_step % update_every_steps == 0:
            beta = min(1.0, per_beta_start + global_step * (1.0 - per_beta_start) / per_beta_frames)

            for _ in range(updates_per_step):
                batch = replay.sample(batch_size, beta)
                if batch is None:
                    break

                idxs, b_s, b_a, b_r, b_ns, b_d, b_w = batch
                (
                    q1_params, q1_opt_st   ,
                    q2_params, q2_opt_st   ,
                    pi_params, pi_opt_st   ,
                    tq1_params, tq2_params ,
                    log_alpha, alpha_opt_st,
                    l_q1, l_q2, l_pi, ent, per_err, a_loss
                ) = update_step(
                    q1_params, q1_opt_st   ,
                    q2_params, q2_opt_st   ,
                    pi_params, pi_opt_st   ,
                    tq1_params, tq2_params ,
                    log_alpha, alpha_opt_st,
                    b_s, b_a, b_r, b_ns, b_d, b_w
                )

                last_l_q1   = float(l_q1)
                last_l_q2   = float(l_q2)
                last_l_pi   = float(l_pi)
                last_ent    = float(ent)
                last_a_loss = float(a_loss)

                ep_ents.append(last_ent)

                per_err_np = np.array(per_err) + 1e-6
                for i in range(batch_size):
                    replay.update(idxs[i], per_err_np[i])

        if done_boundary:
            curr_alpha = float(jnp.exp(log_alpha)) if ALPHA_AUTO else init_alpha
            mean_ent   = float(np.mean(ep_ents)) if len(ep_ents) > 0 else 0.0

            print(f"Ep {episode:4d} | Step {global_step:7d} | Ret {ep_ret:8.1f} | "
                  f"Q1 {last_l_q1:7.3f} | Pi {last_l_pi:7.3f} | "
                  f"Ent {mean_ent:7.3f} | Alpha {curr_alpha:8.4f}")

            if live_plot and (episode % plot_every_episodes == 0):
                plot_eps  .append(episode)
                plot_ent  .append(mean_ent)
                plot_alpha.append(curr_alpha)

                ent_line.set_data(plot_eps, plot_ent)
                tgt_line.set_data(plot_eps, [target_entropy] * len(plot_eps))
                alp_line.set_data(plot_eps, plot_alpha)

                ax1.relim()
                ax1.autoscale_view()

                ax2.relim()
                ax2.autoscale_view()

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            ep_ret  = 0.0
            ep_ents = []
            episode += 1

            obs, _ = env.reset()
            obs    = np.array(obs, dtype=np.float32)

finally:
    env.close()
    if live_plot:
        plt.close()