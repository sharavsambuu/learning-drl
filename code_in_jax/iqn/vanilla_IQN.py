#
# IQN aka Implicit Quantile Networks which is improvements over previous C51 and QR-DQN
#
#   IQN is like making the quantile prediction process "smarter and more efficient." 
#   Instead of always predicting the same fixed set of quantiles (QR-DQN), 
#   IQN learns a function that can generate quantile values on-demand for any quantile 
#   fraction you ask for. This "implicit quantile function" allows for a more flexible, 
#   efficient, and potentially more accurate representation of value distributions in RL. 
#   It's like having a quantile "generator" rather than just a quantile "predictor."
#

import os
import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
from collections import deque


debug_render           = True
debug                  = False
num_episodes           = 500
batch_size             = 64
learning_rate          = 0.001
sync_steps             = 100
memory_length          = 1000
epsilon                = 1.0
epsilon_decay          = 0.001
epsilon_max            = 1.0
epsilon_min            = 0.01
gamma                  = 0.99

plot_every_steps       = 25
render_every_steps     = 10
pause_seconds          = 0.0001

quantile_embedding_dim = 64
N                      = 64
N_prime                = 64
kappa                  = 1.0

N_plot                 = 64
history_length         = 300


tau_plot               = (jnp.arange(N_plot, dtype=jnp.float32) + 0.5) / float(N_plot)
tau_plot_np            = np.array(tau_plot, dtype=np.float32)

q10_idx                = int(0.10 * (N_plot - 1))
q90_idx                = int(0.90 * (N_plot - 1))

N_prio                 = 32
tau_prio               = (jnp.arange(N_prio, dtype=jnp.float32) + 0.5) / float(N_prio)


class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data      = np.zeros(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

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
        idx                   = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx     = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PERMemory:
    e = 0.01
    a = 0.6
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return float((error + self.e) ** self.a)

    def add(self, error, sample):
        p = self._get_priority(float(error))
        self.tree.add(p, sample)

    def sample(self, n):
        batch   = []
        total_p = self.tree.total()
        if total_p <= 0.0:
            return batch

        segment = total_p / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, error):
        p = self._get_priority(float(error))
        self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class IQN(nn.Module):
    n_actions              : int
    quantile_embedding_dim : int

    @nn.compact
    def __call__(self, x, tau):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        state_embedding = nn.relu(x)                                   # (B, 64)

        i_pi            = jnp.arange(1, self.quantile_embedding_dim + 1, dtype=jnp.float32) * jnp.pi
        tau_expanded    = tau[:, :, None]                              # (B, N, 1)
        tau_embedding   = jnp.cos(tau_expanded * i_pi[None, None, :])
        tau_embedding   = nn.Dense(features=64)(tau_embedding)         # (B, N, 64)
        tau_embedding   = nn.relu(tau_embedding)

        state_embedding = state_embedding[:, None, :]                  # (B, 1, 64)
        x               = state_embedding * tau_embedding              # (B, N, 64)
        x               = nn.Dense(features=64)(x)
        x               = nn.relu(x)
        x               = nn.Dense(features=self.n_actions)(x)         # (B, N, A)

        x               = jnp.transpose(x, (0, 2, 1))                  # (B, A, N)
        return x


per_memory = PERMemory(memory_length)

env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n


nn_module   = IQN(n_actions=n_actions, quantile_embedding_dim=quantile_embedding_dim)
dummy_state = jnp.zeros((1,) + state.shape, dtype=jnp.float32)
dummy_tau   = jnp.zeros((1, N), dtype=jnp.float32)
params      = nn_module.init(jax.random.PRNGKey(0), dummy_state, dummy_tau)['params']

nn_model_params        = params
target_nn_model_params = params

optimizer_def          = optax.adam(learning_rate)
optimizer_state        = optimizer_def.init(nn_model_params)


@jax.jit
def inference(params, states, tau):
    return nn_module.apply({'params': params}, states, tau)                     # (B, A, N)


@jax.jit
def quantile_huber_loss_f(pred_quantiles, target_quantiles, tau):
    # pred_quantiles   : (B, N)
    # target_quantiles : (B, N')
    # tau              : (B, N)

    u           = target_quantiles[:, None, :] - pred_quantiles[:, :, None]     # (B, N, N')
    abs_u       = jnp.abs(u)

    quadratic   = jnp.minimum(abs_u, kappa)
    linear      = abs_u - quadratic
    huber_loss  = 0.5 * quadratic**2 + kappa * linear

    indicator   = (u < 0.0).astype(jnp.float32)
    weights     = jnp.abs(tau[:, :, None] - indicator)

    loss        = weights * huber_loss
    return jnp.mean(jnp.sum(loss, axis=(1, 2)))


@jax.jit
def backpropagate(rng_key, optimizer_state, model_params, target_model_params, states, actions, rewards, next_states, dones):
    def loss_fn(params, rng_key):
        rng_key, tau_key, tau_prime_key = jax.random.split(rng_key, 3)

        tau       = jax.random.uniform(tau_key      , (states.shape[0], N      ), minval=0.0, maxval=1.0, dtype=jnp.float32)
        tau_prime = jax.random.uniform(tau_prime_key, (states.shape[0], N_prime), minval=0.0, maxval=1.0, dtype=jnp.float32)

        current_quantiles      = nn_module.apply({'params': params             }, states     , tau      )  # (B, A, N )
        next_quantiles_online  = nn_module.apply({'params': params             }, next_states, tau_prime)  # (B, A, N')
        next_quantiles_target  = nn_module.apply({'params': target_model_params}, next_states, tau_prime)  # (B, A, N')

        next_q_online          = jnp.mean(next_quantiles_online, axis=2)                                   # (B, A)
        best_next_actions      = jnp.argmax(next_q_online, axis=1)                                         # (B,)

        next_target_selected   = next_quantiles_target[jnp.arange(states.shape[0]), best_next_actions]     # (B, N')

        target_u               = rewards[:, None] + gamma * (1.0 - dones[:, None]) * next_target_selected  # (B, N')

        current_selected       = current_quantiles[jnp.arange(states.shape[0]), actions]                   # (B, N)

        loss                   = quantile_huber_loss_f(current_selected, target_u, tau)
        return loss, rng_key

    (loss, new_rng_key), gradients = jax.value_and_grad(loss_fn, has_aux=True)(model_params, rng_key)
    updates, new_optimizer_state   = optimizer_def.update(gradients, optimizer_state, model_params)
    new_model_params               = optax.apply_updates(model_params, updates)
    return new_rng_key, new_optimizer_state, new_model_params, loss


@jax.jit
def td_error_batch_fn(model_params, target_model_params, states, actions, rewards, next_states, dones, tau, tau_prime):
    current_quantiles      = nn_module.apply({'params': model_params       }, states     , tau      )     # (B, A, N)
    next_quantiles_online  = nn_module.apply({'params': model_params       }, next_states, tau_prime)     # (B, A, N')
    next_quantiles_target  = nn_module.apply({'params': target_model_params}, next_states, tau_prime)     # (B, A, N')

    next_q_online          = jnp.mean(next_quantiles_online, axis=2)                                      # (B, A)
    best_next_actions      = jnp.argmax(next_q_online, axis=1)                                            # (B,)

    next_target_selected   = next_quantiles_target[jnp.arange(states.shape[0]), best_next_actions]        # (B, N')
    target_u               = rewards[:, None] + gamma * (1.0 - dones[:, None]) * next_target_selected     # (B, N')

    current_selected       = current_quantiles[jnp.arange(states.shape[0]), actions]                      # (B, N)

    td_error               = jnp.abs(jnp.mean(target_u, axis=1) - jnp.mean(current_selected, axis=1))     # (B,)
    return td_error


if debug_render:
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax_quantile = axes[0]
    ax_stats    = axes[1]

    color_action0 = "tab:blue"
    color_action1 = "tab:orange"
    color_gap     = "tab:gray"

    ax_quantile.set_xlabel("Quantile Value")
    ax_quantile.set_ylabel("Tau")
    ax_quantile.set_title("IQN Quantile Curves (fixed tau grid)")
    ax_quantile.set_ylim(0.0, 1.0)
    ax_quantile.set_xlim(0.0, 500.0)

    zeros_q = np.zeros((N_plot,), dtype=np.float32)
    (line_q0,) = ax_quantile.plot(zeros_q, tau_plot_np, color=color_action0, linewidth=2.0, label="Action 0")
    (line_q1,) = ax_quantile.plot(zeros_q, tau_plot_np, color=color_action1, linewidth=2.0, label="Action 1")
    ax_quantile.legend(loc="upper left")

    ax_stats.set_xlabel("Steps")
    ax_stats.set_ylabel("Value")
    ax_stats.set_title("Mean + Spread (q90-q10) + Gap")
    ax_stats.set_xlim(0, history_length)
    ax_stats.set_ylim(0.0, 500.0)

    steps_hist  = deque(maxlen=history_length)
    mean0_hist  = deque(maxlen=history_length)
    mean1_hist  = deque(maxlen=history_length)
    spr0_hist   = deque(maxlen=history_length)
    spr1_hist   = deque(maxlen=history_length)
    gap_hist    = deque(maxlen=history_length)

    (line_m0,)  = ax_stats.plot([], [], color=color_action0, linewidth=2.0, label="Mean A0")
    (line_m1,)  = ax_stats.plot([], [], color=color_action1, linewidth=2.0, label="Mean A1")
    (line_s0,)  = ax_stats.plot([], [], color=color_action0, linewidth=2.0, alpha=0.45, linestyle="--", label="Spread A0")
    (line_s1,)  = ax_stats.plot([], [], color=color_action1, linewidth=2.0, alpha=0.45, linestyle="--", label="Spread A1")
    (line_g ,)  = ax_stats.plot([], [], color=color_gap    , linewidth=2.0, alpha=0.70, linestyle=":",  label="Gap (best-other)")
    ax_stats.legend(loc="upper left")

    fig.tight_layout()
    fig.show()
    fig.canvas.draw()


rng          = jax.random.PRNGKey(0)
global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)

        while True:
            global_steps += 1

            outputs_plot = None

            if debug_render and (global_steps % plot_every_steps == 0):
                tau_plot_batch = jnp.asarray(np.tile(tau_plot_np[None, :], (1, 1)), dtype=jnp.float32)
                outputs_plot   = inference(nn_model_params, jnp.asarray([state], dtype=jnp.float32), tau_plot_batch)  # (1, A, N_plot)

                try:
                    q0 = np.array(outputs_plot[0][0], dtype=np.float32)
                    q1 = np.array(outputs_plot[0][1], dtype=np.float32)

                    line_q0.set_xdata(q0)
                    line_q1.set_xdata(q1)

                    q_min = float(np.min([q0.min(), q1.min()]))
                    q_max = float(np.max([q0.max(), q1.max()]))
                    q_pad = max(5.0, 0.08 * (q_max - q_min + 1e-6))
                    ax_quantile.set_xlim(q_min - q_pad, q_max + q_pad)

                    m0  = float(q0.mean())
                    m1  = float(q1.mean())
                    s0  = float(q0[q90_idx] - q0[q10_idx])
                    s1  = float(q1[q90_idx] - q1[q10_idx])

                    if m0 >= m1:
                        gap = float(m0 - m1)
                    else:
                        gap = float(m1 - m0)

                    steps_hist.append(global_steps)
                    mean0_hist.append(m0)
                    mean1_hist.append(m1)
                    spr0_hist .append(s0)
                    spr1_hist .append(s1)
                    gap_hist  .append(gap)

                    xs = np.arange(len(steps_hist), dtype=np.float32)

                    line_m0.set_data(xs, np.array(mean0_hist, dtype=np.float32))
                    line_m1.set_data(xs, np.array(mean1_hist, dtype=np.float32))
                    line_s0.set_data(xs, np.array(spr0_hist , dtype=np.float32))
                    line_s1.set_data(xs, np.array(spr1_hist , dtype=np.float32))
                    line_g .set_data(xs, np.array(gap_hist  , dtype=np.float32))

                    ax_stats.set_xlim(0, max(10, len(steps_hist)))

                    all_vals = np.concatenate([
                        np.array(mean0_hist, dtype=np.float32),
                        np.array(mean1_hist, dtype=np.float32),
                        np.array(spr0_hist , dtype=np.float32),
                        np.array(spr1_hist , dtype=np.float32),
                        np.array(gap_hist  , dtype=np.float32),
                    ])
                    y_min = float(all_vals.min())
                    y_max = float(all_vals.max())
                    y_pad = max(5.0, 0.10 * (y_max - y_min + 1e-6))
                    ax_stats.set_ylim(y_min - y_pad, y_max + y_pad)

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(pause_seconds)
                except Exception as e:
                    print(f"Error during plotting: {e}")

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                tau_policy_batch = jnp.asarray(np.tile(tau_plot_np[None, :], (1, 1)), dtype=jnp.float32)
                q_all            = inference(nn_model_params, jnp.asarray([state], dtype=jnp.float32), tau_policy_batch)             # (1, A, N_plot)
                q_values         = jnp.mean(q_all[0], axis=1)                                                                        # (A,)
                action           = int(jnp.argmax(q_values))

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done      = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            # PER initial priority (fixed tau grid, stable)
            tau_prio_batch      = jnp.asarray(np.tile(np.array(tau_prio)[None, :], (1, 1)), dtype=jnp.float32)
            current_prio        = inference(nn_model_params       , jnp.asarray([state    ], dtype=jnp.float32), tau_prio_batch)[0]  # (A, N_prio)
            next_online_prio    = inference(nn_model_params       , jnp.asarray([new_state], dtype=jnp.float32), tau_prio_batch)[0]  # (A, N_prio)
            next_target_prio    = inference(target_nn_model_params, jnp.asarray([new_state], dtype=jnp.float32), tau_prio_batch)[0]  # (A, N_prio)

            best_next_action    = int(jnp.argmax(jnp.mean(next_online_prio, axis=1)))
            target_next_quant   = next_target_prio[best_next_action]                                                                 # (N_prio,)
            u                   = reward + gamma * (1 - int(done)) * target_next_quant

            td_error            = float(jnp.abs(jnp.mean(u) - jnp.mean(current_prio[action])))

            per_memory.add(td_error, (state, action, reward, new_state, float(done)))

            if per_memory.size() >= batch_size:
                batch = per_memory.sample(batch_size)

                states, actions, rewards, next_states, dones = [], [], [], [], []
                idxs = []
                for i in range(batch_size):
                    idxs       .append(batch[i][0]   )
                    states     .append(batch[i][1][0])
                    actions    .append(batch[i][1][1])
                    rewards    .append(batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones      .append(batch[i][1][4])

                rng, step_key = jax.random.split(rng)

                step_key, optimizer_state, nn_model_params, loss = backpropagate(
                    step_key, optimizer_state, nn_model_params, target_nn_model_params,
                    jnp.asarray(states     , dtype=jnp.float32),
                    jnp.asarray(actions    , dtype=jnp.int32  ),
                    jnp.asarray(rewards    , dtype=jnp.float32),
                    jnp.asarray(next_states, dtype=jnp.float32),
                    jnp.asarray(dones      , dtype=jnp.float32),
                )

                tau_prio_b         = jnp.asarray(np.tile(np.array(tau_prio)[None, :], (batch_size, 1)), dtype=jnp.float32)
                td_errors_batch    = td_error_batch_fn(
                    nn_model_params,
                    target_nn_model_params,
                    jnp.asarray(states     , dtype=jnp.float32),
                    jnp.asarray(actions    , dtype=jnp.int32  ),
                    jnp.asarray(rewards    , dtype=jnp.float32),
                    jnp.asarray(next_states, dtype=jnp.float32),
                    jnp.asarray(dones      , dtype=jnp.float32),
                    tau_prio_b,
                    tau_prio_b
                )
                td_errors_batch = np.array(td_errors_batch, dtype=np.float32)

                for i in range(batch_size):
                    per_memory.update(idxs[i], float(td_errors_batch[i]))

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_nn_model_params = nn_model_params

            if debug_render and (global_steps % render_every_steps == 0):
                env.render()

            if done:
                print("{} episode, reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
