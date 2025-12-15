#
# SAC (Continuous) 
#
#    - Continuous SAC with Gaussian Policy (Reparameterization trick + tanh squashing)
#    - Correct action scaling (policy outputs env-range actions; critics always see env-range actions)
#    - Twin Q networks + clipped double Q target
#    - Prioritized Experience Replay (PER) SumTree (new samples start at max priority)
#    - Fixed entropy alpha (simple + stable; no autotune here)
#    - Single compiled JAX update step for performance
#    - Correct boundary handling:
#         terminated  -> done_term  (mask bootstrap)
#         truncated   -> boundary reset only (bootstrap allowed)
#    - Trio plots:
#         - Episode reward
#         - Critic loss (avg)
#         - Policy entropy estimate = E[-log_pi(a|s)]
#
#

import random
import jax
import optax
import numpy             as np
import gymnasium         as gym
import flax.linen        as nn
from   jax               import numpy as jnp
import matplotlib.pyplot as plt


# Hyperparameters
env_name        = "Pendulum-v1"
debug_render    = True

num_episodes    = 500
learning_rate   = 3e-4
gamma           = 0.99
tau             = 0.005
entropy_alpha   = 0.2
batch_size      = 256
memory_length   = 1000000

max_grad_norm   = 10.0

per_alpha       = 0.6
per_beta_start  = 0.4
per_beta_frames = 100000
per_epsilon     = 1e-6

live_plot       = True


class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 4))

        self.axs[0].set_title("Episode Reward")
        self.axs[0].set_xlabel("Episode")
        self.line_rew, = self.axs[0].plot([], [], color="green")
        self.axs[0].grid(True, alpha=0.3)

        self.axs[1].set_title("Critic Loss (Avg)")
        self.axs[1].set_xlabel("Step")
        self.line_loss, = self.axs[1].plot([], [], color="red", alpha=0.6)
        self.axs[1].grid(True, alpha=0.3)
        self.axs[1].set_yscale("log")

        self.axs[2].set_title("Entropy: E[-log_pi(a|s)]")
        self.axs[2].set_xlabel("Step")
        self.line_ent, = self.axs[2].plot([], [], color="blue")
        self.axs[2].grid(True, alpha=0.3)

        self.episodes   = []
        self.rewards    = []

        self.steps      = []
        self.losses     = []
        self.entropies  = []

        plt.tight_layout()
        plt.show(block=False)

    def update(self, episode, reward, step, avg_loss, avg_ent):
        self.episodes.append(episode)
        self.rewards .append(reward)
        self.line_rew.set_data(self.episodes, self.rewards)
        self.axs[0].relim()
        self.axs[0].autoscale_view()

        if step is not None:
            self.steps     .append(step)
            self.losses    .append(avg_loss)
            self.entropies .append(avg_ent)

            self.line_loss.set_data(self.steps, self.losses)
            self.line_ent .set_data(self.steps, self.entropies)

            self.axs[1].relim()
            self.axs[1].autoscale_view()
            self.axs[2].relim()
            self.axs[2].autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()


class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float32)
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

        total_p = self.tree.total()
        if total_p <= 0.0:
            return None

        batch, idxs, priorities = [], [], []
        segment = total_p / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if data is None:
                return None
            batch     .append(data)
            idxs      .append(idx)
            priorities.append(p)

        probs   = np.array(priorities, dtype=np.float32) / total_p
        weights = (self.tree.size * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-8)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            idxs,
            np.array(states     , dtype=np.float32),
            np.array(actions    , dtype=np.float32),
            np.array(rewards    , dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones      , dtype=np.float32),
            np.array(weights    , dtype=np.float32),
        )

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        if p > self.max_p:
            self.max_p = float(p)

    def __len__(self):
        return self.tree.size



class SoftQNetwork(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class GaussianPolicyNetwork(nn.Module):
    action_dim : int
    log_std_min: float = -20.
    log_std_max: float = 2.

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(256)(state)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean    = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


env         = gym.make(env_name, render_mode="human" if debug_render else None)
state_dim   = env.observation_space.shape[0]
action_dim  = env.action_space.shape[0]

act_high    = np.array(env.action_space.high, dtype=np.float32)
act_low     = np.array(env.action_space.low , dtype=np.float32)

act_scale   = (act_high - act_low) / 2.0
act_bias    = (act_high + act_low) / 2.0

act_scale_j = jnp.array(act_scale, dtype=jnp.float32)
act_bias_j  = jnp.array(act_bias , dtype=jnp.float32)


soft_q_module_1 = SoftQNetwork()
soft_q_module_2 = SoftQNetwork()
policy_module   = GaussianPolicyNetwork(action_dim=action_dim)

rng = jax.random.PRNGKey(42)
rng, k1, k2, k3 = jax.random.split(rng, 4)

dummy_s = jnp.zeros((1, state_dim), dtype=jnp.float32)
dummy_a = jnp.zeros((1, action_dim), dtype=jnp.float32)

q1_params  = soft_q_module_1.init(k1, dummy_s, dummy_a)["params"]
q2_params  = soft_q_module_2.init(k2, dummy_s, dummy_a)["params"]
tq1_params = q1_params
tq2_params = q2_params
pi_params  = policy_module.init(k3, dummy_s)["params"]

optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(learning_rate, eps=1e-5)
)

q1_opt_st = optimizer.init(q1_params)
q2_opt_st = optimizer.init(q2_params)
pi_opt_st = optimizer.init(pi_params)

replay_memory = PrioritizedReplayMemory(memory_length)


@jax.jit
def soft_update(target, source, tau):
    return jax.tree_util.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)


@jax.jit
def sample_action(rng, policy_params, state, act_scale_j, act_bias_j):
    mean, log_std = policy_module.apply({"params": policy_params}, state)
    std           = jnp.exp(log_std)

    normal        = jax.random.normal(rng, mean.shape)
    z             = mean + std * normal

    tanh_a        = jnp.tanh(z)
    action        = tanh_a * act_scale_j + act_bias_j

    # Base log N(z|mean,std)
    log_prob = -0.5 * (((z - mean) / (std + 1e-6)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))

    # Stable tanh correction: log(1 - tanh(z)^2) = 2*(log(2) - z - softplus(-2z))
    log_prob -= 2.0 * (jnp.log(2.0) - z - jax.nn.softplus(-2.0 * z))

    log_prob  = log_prob.sum(axis=-1, keepdims=True)

    # Scaling/bias change-of-variables is a constant shift in log_prob (keeps objective "exact")
    log_prob -= jnp.sum(jnp.log(act_scale_j + 1e-6))

    return action, log_prob


@jax.jit
def update_step(
    rng       ,
    q1_params , q1_opt_st ,
    q2_params , q2_opt_st ,
    pi_params , pi_opt_st ,
    tq1_params, tq2_params,
    states, actions, rewards, next_states, dones, is_weights,
    act_scale_j, act_bias_j
):
    rng, k_target, k_policy = jax.random.split(rng, 3)

    # Q-Function Update
    next_actions, next_logp = sample_action(k_target, pi_params, next_states, act_scale_j, act_bias_j)

    tq1 = soft_q_module_1.apply({"params": tq1_params}, next_states, next_actions)
    tq2 = soft_q_module_2.apply({"params": tq2_params}, next_states, next_actions)

    tq_min   = jnp.minimum(tq1, tq2) - entropy_alpha * next_logp

    rewards  = rewards    [:, None]
    dones    = dones      [:, None]
    is_w     = is_weights [:, None]

    target_q = rewards + (1.0 - dones) * gamma * tq_min

    def q_loss_fn(p, apply_fn):
        q_pred = apply_fn({"params": p}, states, actions)
        td     = target_q - q_pred
        loss   = jnp.mean(is_w * jnp.square(td))
        return loss, td

    (q1_loss, td1), g1 = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, soft_q_module_1.apply)
    u1, q1_opt_st      = optimizer.update(g1, q1_opt_st, q1_params)
    q1_params          = optax.apply_updates(q1_params, u1)

    (q2_loss, td2), g2 = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, soft_q_module_2.apply)
    u2, q2_opt_st      = optimizer.update(g2, q2_opt_st, q2_params)
    q2_params          = optax.apply_updates(q2_params, u2)

    # Policy Update
    def pi_loss_fn(p):
        curr_actions, curr_logp = sample_action(k_policy, p, states, act_scale_j, act_bias_j)

        q1 = soft_q_module_1.apply({"params": q1_params}, states, curr_actions)
        q2 = soft_q_module_2.apply({"params": q2_params}, states, curr_actions)
        q_min = jnp.minimum(q1, q2)

        loss     = jnp.mean(entropy_alpha * curr_logp - q_min)
        entropy  = -jnp.mean(curr_logp)
        return loss, entropy

    (pi_loss, ent), g_pi = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
    u_pi, pi_opt_st      = optimizer.update(g_pi, pi_opt_st, pi_params)
    pi_params            = optax.apply_updates(pi_params, u_pi)

    # Soft Update Targets
    tq1_params = soft_update(tq1_params, q1_params, tau)
    tq2_params = soft_update(tq2_params, q2_params, tau)

    per_err    = 0.5 * (jnp.abs(td1) + jnp.abs(td2))
    q_loss_avg = 0.5 * (q1_loss + q2_loss)

    return (
        rng,
        q1_params, q1_opt_st,
        q2_params, q2_opt_st,
        pi_params, pi_opt_st,
        tq1_params, tq2_params,
        q_loss_avg, pi_loss, ent, per_err
    )


global_step = 0
per_beta    = per_beta_start

plotter     = LivePlotter() if live_plot else None

try:
    for episode in range(num_episodes):
        state, _ = env.reset()
        state    = np.array(state, dtype=np.float32)

        done     = False
        ep_ret   = 0.0

        ep_q_loss = []
        ep_ent    = []

        while not done:
            global_step += 1

            rng, a_rng = jax.random.split(rng)

            if global_step < 1000:  # Warmup
                action = env.action_space.sample()
            else:
                action_jax, _ = sample_action(a_rng, pi_params, state[None, :], act_scale_j, act_bias_j)
                action = np.array(action_jax[0], dtype=np.float32)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done_boundary = bool(terminated or truncated)
            done_term     = float(terminated)

            next_state = np.array(next_state, dtype=np.float32)
            ep_ret    += float(reward)

            replay_memory.push(state, action, float(reward), next_state, done_term)

            state = next_state
            done  = done_boundary

            if len(replay_memory) > batch_size:
                per_beta = min(1.0, per_beta_start + global_step * (1.0 - per_beta_start) / per_beta_frames)

                batch = replay_memory.sample(batch_size, per_beta)
                if batch is not None:
                    idxs, b_s, b_a, b_r, b_ns, b_d, b_w = batch
                    (
                        rng       ,
                        q1_params , q1_opt_st ,
                        q2_params , q2_opt_st ,
                        pi_params , pi_opt_st ,
                        tq1_params, tq2_params,
                        l_q, l_pi, l_ent, per_err
                    ) = update_step(
                        rng      ,
                        q1_params, q1_opt_st  ,
                        q2_params, q2_opt_st  ,
                        pi_params, pi_opt_st  ,
                        tq1_params, tq2_params,
                        b_s, b_a, b_r, b_ns, b_d, b_w,
                        act_scale_j, act_bias_j
                    )

                    per_err_np = np.array(per_err).reshape(-1)
                    for i in range(batch_size):
                        replay_memory.update(idxs[i], per_err_np[i])

                    ep_q_loss.append(float(l_q))
                    ep_ent   .append(float(l_ent))

        avg_loss = float(np.mean(ep_q_loss)) if ep_q_loss else 0.0
        avg_ent  = float(np.mean(ep_ent   )) if ep_ent    else 0.0

        print(f"Ep {episode:4d} | Step {global_step:7d} | Ret {ep_ret:8.1f} | "
              f"Q-Loss {avg_loss:9.4f} | Ent {avg_ent:8.3f}")

        if live_plot:
            plotter.update(episode, ep_ret, global_step, avg_loss, avg_ent)

finally:
    env.close()
    if live_plot:
        plotter.close()
