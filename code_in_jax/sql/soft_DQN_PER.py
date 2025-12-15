#
# Soft Q-Learning (SQL) + PER + Target Network + Live Plots 
#
# Implemented:
#   - Soft Bellman backup via log-sum-exp value: V(s) = tau * logmeanexp(Q(s,a)/tau)
#   - Soft policy: pi(a|s) = softmax(Q(s,a)/tau)
#   - Prioritized Experience Replay (PER) using a SumTree (tracks n_entries)
#   - Target network sync every sync_steps
#   - Live matplotlib plots:
#       1) Q(s,a) bars
#       2) pi(a|s) bars
#       3) V(s) history
#       4) Bellman target vs Q(s,a) history
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


debug_render       = True
debug              = False
num_episodes       = 500
batch_size         = 64
learning_rate      = 0.001
sync_steps         = 100
memory_length      = 4000
tau                = 0.005
gamma              = 0.99

plot_every_steps   = 25
render_every_steps = 10
pause_seconds      = 0.0001

history_len        = 300


class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data      = np.zeros(capacity, dtype=object)
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
        return (idx, float(self.tree[idx]), self.data[dataIdx])


class PERMemory:
    e = 0.01
    a = 0.6
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return float((error + self.e) ** self.a)

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch   = []
        total_p = self.tree.total()
        if total_p <= 0 or self.size() <= 0:
            return batch

        segment = total_p / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            picked = None
            for _ in range(16):
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if isinstance(data, tuple) and len(data) >= 5:
                    picked = (idx, data)
                    break

            if picked is None:
                for _ in range(64):
                    s = random.uniform(0.0, total_p)
                    (idx, p, data) = self.tree.get(s)
                    if isinstance(data, tuple) and len(data) >= 5:
                        picked = (idx, data)
                        break

            if picked is not None:
                batch.append(picked)

        return batch

    def update(self, idx, error):
        p = self._get_priority(float(error))
        self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class SoftQNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        q_values = nn.Dense(features=self.n_actions)(x)
        return q_values


env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n

dqn_module  = SoftQNetwork(n_actions=n_actions)
dummy_input = jnp.zeros(state.shape, dtype=jnp.float32)
params      = dqn_module.init(jax.random.PRNGKey(0), dummy_input)

q_network_params        = params['params']
target_q_network_params = params['params']

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(q_network_params)

per_memory = PERMemory(memory_length)


@jax.jit
def inference(params, input_batch):
    return dqn_module.apply({'params': params}, input_batch)

@jax.jit
def soft_value_from_q(q_values):
    v = jax.scipy.special.logsumexp(q_values / tau, axis=-1) - jnp.log(q_values.shape[-1])
    return tau * v

@jax.jit
def policy_probs_from_q(q_values):
    return jax.nn.softmax(q_values / tau, axis=-1)

def calculate_td_error(q_value_vec, target_v_value, action, reward):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value = jnp.sum(one_hot_actions * q_value_vec, axis=-1)
    td_error = reward + gamma * target_v_value - q_value
    return jnp.abs(td_error)

calculate_td_error_vmap = jax.vmap(calculate_td_error, in_axes=(0, 0, 0, 0), out_axes=0)

@jax.jit
def td_error_batch(q_network_params, target_q_network_params, batch):
    predicted_q_values = dqn_module.apply({'params': q_network_params}, batch[0])
    target_q_values    = dqn_module.apply({'params': target_q_network_params}, batch[3])
    target_v_value     = soft_value_from_q(target_q_values)
    return calculate_td_error_vmap(predicted_q_values, target_v_value, batch[1], batch[2])

def soft_q_learning_loss(q_value_vec, target_v_value, action, reward, done):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value   = jnp.sum(one_hot_actions * q_value_vec, axis=-1)
    td_target = reward + gamma * target_v_value * (1.0 - done)
    td_error  = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

soft_q_learning_loss_vmap = jax.vmap(soft_q_learning_loss, in_axes=(0, 0, 0, 0, 0), out_axes=0)

@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch):
    def loss_fn(params):
        predicted_q_values = dqn_module.apply({'params': params}, batch[0])
        target_q_values    = dqn_module.apply({'params': target_q_network_params}, batch[3])
        target_v_value     = soft_value_from_q(target_q_values)
        losses = soft_q_learning_loss_vmap(
            predicted_q_values,
            target_v_value,
            batch[1],
            batch[2],
            batch[4]
        )
        return jnp.mean(losses)

    loss, gradients    = jax.value_and_grad(loss_fn)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params   = optax.apply_updates(q_network_params, updates)
    current_td_errors  = td_error_batch(q_network_params, target_q_network_params, batch)
    return q_network_params, opt_state, loss, current_td_errors


v_hist   = deque(maxlen=history_len)
q_hist   = deque(maxlen=history_len)
tgt_hist = deque(maxlen=history_len)

if debug_render:
    plt.ion()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    ax_q   = axes[0]
    ax_pi  = axes[1]
    ax_v   = axes[2]
    ax_bt  = axes[3]

    ax_q.set_title("Q(s,a)")
    ax_pi.set_title("pi(a|s)")
    ax_v.set_title("V(s) history")
    ax_bt.set_title("Target vs Q(s,a)")

    ax_q.set_ylim(-5.0, 5.0)
    ax_pi.set_ylim(0.0, 1.0)

    ax_v.set_xlim(0, history_len)
    ax_bt.set_xlim(0, history_len)

    x_actions = np.arange(n_actions, dtype=np.int32)

    q_bars  = ax_q.bar(x_actions, np.zeros(n_actions, dtype=np.float32), alpha=0.85)
    pi_bars = ax_pi.bar(x_actions, np.zeros(n_actions, dtype=np.float32), alpha=0.85)

    v_line,   = ax_v.plot(np.zeros(history_len, dtype=np.float32))
    q_line,   = ax_bt.plot(np.zeros(history_len, dtype=np.float32), label="Q(s,a)")
    tgt_line, = ax_bt.plot(np.zeros(history_len, dtype=np.float32), label="Target")

    ax_bt.legend(loc="upper left")

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
                outputs_plot = inference(q_network_params, jnp.asarray([state], dtype=jnp.float32))[0]

                try:
                    q_np  = np.array(outputs_plot, dtype=np.float32)
                    pi_np = np.array(policy_probs_from_q(outputs_plot), dtype=np.float32)
                    v_val = float(soft_value_from_q(outputs_plot))

                    for bar, h in zip(q_bars, q_np):
                        bar.set_height(float(h))
                    for bar, h in zip(pi_bars, pi_np):
                        bar.set_height(float(h))

                    v_hist.append(v_val)

                    v_arr = np.zeros(history_len, dtype=np.float32)
                    if len(v_hist) > 0:
                        v_arr[-len(v_hist):] = np.array(v_hist, dtype=np.float32)
                    v_line.set_ydata(v_arr)

                    q_arr = np.zeros(history_len, dtype=np.float32)
                    t_arr = np.zeros(history_len, dtype=np.float32)
                    if len(q_hist) > 0:
                        q_arr[-len(q_hist):] = np.array(q_hist, dtype=np.float32)
                    if len(tgt_hist) > 0:
                        t_arr[-len(tgt_hist):] = np.array(tgt_hist, dtype=np.float32)

                    q_line.set_ydata(q_arr)
                    tgt_line.set_ydata(t_arr)

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(pause_seconds)
                except Exception as e:
                    print(f"Error during plotting: {e}")

            if global_steps < 1000:
                action = env.action_space.sample()
            else:
                if outputs_plot is None:
                    outputs_plot = inference(q_network_params, jnp.asarray([state], dtype=jnp.float32))[0]
                pi = policy_probs_from_q(outputs_plot)
                action = jax.random.choice(jax.random.PRNGKey(global_steps), n_actions, p=pi)
                action = int(action)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            q_curr = inference(q_network_params, jnp.asarray([state], dtype=jnp.float32))[0]
            v_next = soft_value_from_q(inference(target_q_network_params, jnp.asarray([new_state], dtype=jnp.float32))[0])

            q_sa   = float(q_curr[action])
            target = float(reward + gamma * (1.0 - float(done)) * v_next)

            q_hist.append(q_sa)
            tgt_hist.append(target)

            temporal_difference = float(td_error_batch(
                q_network_params,
                target_q_network_params,
                (
                    jnp.asarray([state], dtype=jnp.float32),
                    jnp.asarray([action], dtype=jnp.int32),
                    jnp.asarray([reward], dtype=jnp.float32),
                    jnp.asarray([new_state], dtype=jnp.float32),
                )
            )[0])

            per_memory.add(temporal_difference, (state, action, reward, new_state, float(done)))

            if per_memory.size() >= batch_size:
                batch = per_memory.sample(batch_size)
                if len(batch) >= batch_size:
                    idxs, segment_data = zip(*batch)

                    states, actions, rewards, next_states, dones = [], [], [], [], []
                    for data in segment_data:
                        states     .append(data[0])
                        actions    .append(data[1])
                        rewards    .append(data[2])
                        next_states.append(data[3])
                        dones      .append(data[4])

                    q_network_params, opt_state, loss, new_td_errors = train_step(
                        q_network_params,
                        target_q_network_params,
                        opt_state,
                        (
                            jnp.asarray(states     , dtype=jnp.float32),
                            jnp.asarray(actions    , dtype=jnp.int32  ),
                            jnp.asarray(rewards    , dtype=jnp.float32),
                            jnp.asarray(next_states, dtype=jnp.float32),
                            jnp.asarray(dones      , dtype=jnp.float32),
                        )
                    )

                    new_td_errors_np = np.array(new_td_errors, dtype=np.float32)
                    for i in range(batch_size):
                        per_memory.update(idxs[i], float(new_td_errors_np[i]))

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_q_network_params = q_network_params

            if debug_render and (global_steps % render_every_steps == 0):
                env.render()

            if done:
                print(f"{episode} - Total reward: {sum(episode_rewards)}")
                break

finally:
    env.close()
