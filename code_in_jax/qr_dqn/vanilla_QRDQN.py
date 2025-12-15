#
# QR-DQN aka Quantile Regression DQN which is improvements over C51
#
#   QR-DQN is like upgrading from a histogram-based view of value distributions (C51) to a 
#   more flexible and adaptable quantile-based view. By learning to predict quantiles directly 
#   using quantile regression, QR-DQN aims to capture a richer and more accurate representation 
#   of the uncertainty and shape of the value distribution, potentially leading to 
#   improved RL performance.
#
#   Panel A: Quantile curves (both actions)
#   Panel B: Mean + Spread over time (both actions)
#
#

import os
import random
import math
import matplotlib
matplotlib.use('TkAgg')  # try 'QtAgg' if TkAgg doesn't work
import matplotlib.pyplot as plt
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
from collections import deque


debug_render       = True
debug              = False
num_episodes       = 500
batch_size         = 64
learning_rate      = 0.001
sync_steps         = 100
memory_length      = 1000
epsilon            = 1.0
epsilon_decay      = 0.001
epsilon_max        = 1.0
epsilon_min        = 0.01
gamma              = 0.99

plot_every_steps   = 25
render_every_steps = 10
pause_seconds      = 0.0001

n_quantiles        = 50
kappa              = 1.0

history_length     = 300


tau                = jnp.linspace(0.0, 1.0, n_quantiles + 1)
tau_hat            = (tau[:-1] + tau[1:]) / 2.0
tau_hat_np         = np.array(tau_hat, dtype=np.float32)

q10_idx            = int(0.10 * (n_quantiles - 1))
q90_idx            = int(0.90 * (n_quantiles - 1))


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
        return self.tree[0]

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
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch   = []
        total_p = self.tree.total()
        if total_p <= 0:
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
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class QRDQN(nn.Module):
    n_actions   : int
    n_quantiles : int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions * self.n_quantiles)(x)
        x = jnp.reshape(x, (-1, self.n_actions, self.n_quantiles))
        return x


per_memory = PERMemory(memory_length)

env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n


nn_module              = QRDQN(n_actions=n_actions, n_quantiles=n_quantiles)
dummy_input            = jnp.zeros(state.shape, dtype=jnp.float32)
params                 = nn_module.init(jax.random.PRNGKey(0), dummy_input)['params']

nn_model_params        = params
target_nn_model_params = params

optimizer_def          = optax.adam(learning_rate)
optimizer_state        = optimizer_def.init(nn_model_params)


@jax.jit
def inference(params, input_batch):
    return nn_module.apply({'params': params}, input_batch)


@jax.jit
def quantile_huber_loss_f(predicted_quantiles, target_quantiles):
    u = jnp.expand_dims(target_quantiles, axis=-1) - jnp.expand_dims(predicted_quantiles, axis=0)

    abs_u      = jnp.abs(u)
    quadratic  = jnp.minimum(abs_u, kappa)
    linear     = abs_u - quadratic
    huber_loss = 0.5 * quadratic**2 + kappa * linear

    indicator  = (u < 0).astype(jnp.float32)
    weights    = jnp.abs(jnp.expand_dims(tau_hat, axis=1) - indicator)

    loss = weights * huber_loss
    return jnp.sum(loss)


@jax.jit
def td_error_batch_fn(model_params, target_model_params, states, actions, rewards, next_states, dones):
    predicted_quantiles = nn_module.apply({'params': model_params}, states)
    target_quantiles    = nn_module.apply({'params': target_model_params}, next_states)

    target_q_values     = jnp.mean(target_quantiles, axis=2)
    best_next_actions   = jnp.argmax(target_q_values, axis=1)

    target_best         = target_quantiles[jnp.arange(states.shape[0]), best_next_actions]
    u                   = rewards[:, None] + gamma * (1 - dones[:, None]) * target_best

    predicted_actions   = predicted_quantiles[jnp.arange(states.shape[0]), actions]
    td_errors           = jnp.mean(jnp.abs(u - predicted_actions), axis=1)
    return td_errors


@jax.jit
def backpropagate(optimizer_state, model_params, target_model_params, states, actions, rewards, next_states, dones):
    def loss_fn(params):
        predicted_quantiles = nn_module.apply({'params': params}, states)
        target_quantiles    = nn_module.apply({'params': target_model_params}, next_states)

        target_q_values     = jnp.mean(target_quantiles, axis=2)
        best_next_actions   = jnp.argmax(target_q_values, axis=1)

        target_best         = target_quantiles[jnp.arange(batch_size), best_next_actions]
        u                   = rewards[:, None] + gamma * (1 - dones[:, None]) * target_best

        predicted_actions   = predicted_quantiles[jnp.arange(batch_size), actions]

        losses = jax.vmap(quantile_huber_loss_f, in_axes=(0, 0))(predicted_actions, u)
        return jnp.mean(losses)

    loss, gradients              = jax.value_and_grad(loss_fn)(model_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, model_params)
    new_model_params             = optax.apply_updates(model_params, updates)
    return new_optimizer_state, new_model_params, loss


if debug_render:
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax_quantile = axes[0]
    ax_stats    = axes[1]

    color_action0 = "tab:blue"
    color_action1 = "tab:orange"

    ax_quantile.set_xlabel("Quantile Value")
    ax_quantile.set_ylabel("Tau")
    ax_quantile.set_title("Quantile Curves (both actions)")
    ax_quantile.set_ylim(0.0, 1.0)
    ax_quantile.set_xlim(0.0, 500.0)

    zeros_q = np.zeros((n_quantiles,), dtype=np.float32)
    (line_q0,) = ax_quantile.plot(zeros_q, tau_hat_np, color=color_action0, linewidth=2.0, label="Action 0")
    (line_q1,) = ax_quantile.plot(zeros_q, tau_hat_np, color=color_action1, linewidth=2.0, label="Action 1")
    ax_quantile.legend(loc="upper left")

    ax_stats.set_xlabel("Steps")
    ax_stats.set_ylabel("Value")
    ax_stats.set_title("Mean + Spread (q90-q10)")
    ax_stats.set_xlim(0, history_length)
    ax_stats.set_ylim(0.0, 500.0)

    steps_hist  = deque(maxlen=history_length)
    mean0_hist  = deque(maxlen=history_length)
    mean1_hist  = deque(maxlen=history_length)
    spr0_hist   = deque(maxlen=history_length)
    spr1_hist   = deque(maxlen=history_length)

    (line_m0,)  = ax_stats.plot([], [], color=color_action0, linewidth=2.0, label="Mean A0")
    (line_m1,)  = ax_stats.plot([], [], color=color_action1, linewidth=2.0, label="Mean A1")
    (line_s0,)  = ax_stats.plot([], [], color=color_action0, linewidth=2.0, alpha=0.45, linestyle="--", label="Spread A0")
    (line_s1,)  = ax_stats.plot([], [], color=color_action1, linewidth=2.0, alpha=0.45, linestyle="--", label="Spread A1")
    ax_stats.legend(loc="upper left")

    fig.tight_layout()
    fig.show()
    fig.canvas.draw()


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
                outputs_plot = inference(nn_model_params, jnp.asarray([state], dtype=jnp.float32))  # (1, A, N)

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

                    steps_hist.append(global_steps)
                    mean0_hist.append(m0)
                    mean1_hist.append(m1)
                    spr0_hist .append(s0)
                    spr1_hist .append(s1)

                    xs = np.arange(len(steps_hist), dtype=np.float32)

                    line_m0.set_data(xs, np.array(mean0_hist, dtype=np.float32))
                    line_m1.set_data(xs, np.array(mean1_hist, dtype=np.float32))
                    line_s0.set_data(xs, np.array(spr0_hist , dtype=np.float32))
                    line_s1.set_data(xs, np.array(spr1_hist , dtype=np.float32))

                    ax_stats.set_xlim(0, max(10, len(steps_hist)))

                    all_vals = np.concatenate([
                        np.array(mean0_hist, dtype=np.float32),
                        np.array(mean1_hist, dtype=np.float32),
                        np.array(spr0_hist , dtype=np.float32),
                        np.array(spr1_hist , dtype=np.float32),
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
                if outputs_plot is None:
                    outputs_plot = inference(nn_model_params, jnp.asarray([state], dtype=jnp.float32))
                q_values = jnp.mean(outputs_plot[0], axis=1)
                action   = int(jnp.argmax(q_values))

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            predicted_current = inference(nn_model_params       , jnp.asarray([state    ], dtype=jnp.float32))[0]
            target_next       = inference(target_nn_model_params, jnp.asarray([new_state], dtype=jnp.float32))[0]

            best_next_action  = jnp.argmax(jnp.mean(target_next, axis=1))
            u                 = reward + gamma * (1 - int(done)) * target_next[best_next_action]
            td_error          = jnp.mean(jnp.abs(u - predicted_current[action]))

            per_memory.add(float(td_error), (state, action, reward, new_state, int(done)))

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

                optimizer_state, nn_model_params, loss = backpropagate(
                    optimizer_state, nn_model_params, target_nn_model_params,
                    jnp.asarray(states     , dtype=jnp.float32),
                    jnp.asarray(actions    , dtype=jnp.int32  ),
                    jnp.asarray(rewards    , dtype=jnp.float32),
                    jnp.asarray(next_states, dtype=jnp.float32),
                    jnp.asarray(dones      , dtype=jnp.float32)
                )

                td_errors_batch = td_error_batch_fn(
                    nn_model_params,
                    target_nn_model_params,
                    jnp.asarray(states     , dtype=jnp.float32),
                    jnp.asarray(actions    , dtype=jnp.int32  ),
                    jnp.asarray(rewards    , dtype=jnp.float32),
                    jnp.asarray(next_states, dtype=jnp.float32),
                    jnp.asarray(dones      , dtype=jnp.float32)
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
