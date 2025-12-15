#
# Dueling DQN + Prioritized Experience Replay (PER)  
#

import os
import random
import math
import gymnasium   as gym
from collections import deque

import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax

debug_render  = True
debug         = False
num_episodes  = 500
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99 # discount factor


class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2*capacity - 1, dtype=np.float32)
        self.data      = np.zeros(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent             = (idx - 1) // 2
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
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return float(self.tree[0])

    def add(self, p, data):
        idx                   = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write           += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        p = float(p)
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
    beta_start  = 0.4
    beta_frames = 100000
    epsilon     = 1e-8

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.beta = self.beta_start

    def _get_priority(self, error):
        return (float(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch   = []
        idxs    = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment*i
            b = segment*(i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)

        priorities = np.array([self.tree.tree[idx] for idx in idxs], dtype=np.float32)
        total      = self.tree.total() + self.epsilon
        probs      = priorities / total

        is_weights = np.power(self.tree.n_entries * probs + self.epsilon, -self.beta)
        is_weights /= (is_weights.max() + self.epsilon)

        return idxs, batch, is_weights.astype(np.float32)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def update_beta(self, frame_idx):
        fraction  = min(float(frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)


class DuelingQNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        feature_layer = nn.Dense(features=64)(x)
        feature_layer = nn.relu(feature_layer)
        feature_layer = nn.Dense(features=64)(feature_layer)
        feature_layer = nn.relu(feature_layer)

        value_stream = nn.Dense(features=64)(feature_layer)
        value_stream = nn.relu(value_stream)
        value        = nn.Dense(features=1)(value_stream)

        advantage_stream = nn.Dense(features=64)(feature_layer)
        advantage_stream = nn.relu(advantage_stream)
        advantage        = nn.Dense(features=self.n_actions)(advantage_stream)

        advantage_mean = jnp.mean(advantage, axis=-1, keepdims=True)
        q_values       = value + (advantage - advantage_mean)
        return q_values


env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n

dqn_module              = DuelingQNetwork(n_actions=n_actions)
dummy_input             = jnp.zeros(state.shape, dtype=jnp.float32)
params                  = dqn_module.init(jax.random.PRNGKey(0), dummy_input)
q_network_params        = params['params']
target_q_network_params = params['params']

optimizer               = optax.adam(learning_rate)
opt_state               = optimizer.init(q_network_params)

per_memory              = PERMemory(memory_length)


@jax.jit
def policy(params, x):
    predicted_q_values = dqn_module.apply({'params': params}, x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


@jax.jit
def td_error_batch(q_network_params, target_q_network_params, batch):
    states, actions, rewards, next_states, dones = batch

    predicted_q_values   = dqn_module.apply({'params': q_network_params}, states)
    q_value              = jnp.take_along_axis(predicted_q_values, actions[:, None], axis=1).squeeze(1)

    target_next_q_values = dqn_module.apply({'params': target_q_network_params}, next_states)
    target_next_q_max    = jnp.max(target_next_q_values, axis=1)

    td_target = rewards + gamma * target_next_q_max * (1.0 - dones)
    td_error  = td_target - q_value

    return jnp.abs(td_error)


@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch, is_weights):
    states, actions, rewards, next_states, dones = batch
    is_weights = is_weights.astype(jnp.float32)

    def loss_fn(params):
        predicted_q_values   = dqn_module.apply({'params': params}, states)
        q_value              = jnp.take_along_axis(predicted_q_values, actions[:, None], axis=1).squeeze(1)

        target_next_q_values = dqn_module.apply({'params': target_q_network_params}, next_states)
        target_next_q_max    = jnp.max(target_next_q_values, axis=1)

        td_target = rewards + gamma * target_next_q_max * (1.0 - dones)
        td_error  = jax.lax.stop_gradient(td_target) - q_value

        loss = jnp.mean(is_weights * jnp.square(td_error))
        return loss

    loss, gradients    = jax.value_and_grad(loss_fn)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params   = optax.apply_updates(q_network_params, updates)

    current_td_errors = td_error_batch(
        q_network_params       ,
        target_q_network_params,
        (states, actions, rewards, next_states, dones)
    )

    return q_network_params, opt_state, loss, current_td_errors


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)

        while True:
            global_steps = global_steps + 1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(q_network_params, jnp.asarray(state, dtype=jnp.float32))
                action = int(action)
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            temporal_difference = float(td_error_batch(
                q_network_params,
                target_q_network_params,
                (
                    jnp.asarray([state      ], dtype=jnp.float32),
                    jnp.asarray([action     ], dtype=jnp.int32  ),
                    jnp.asarray([reward     ], dtype=jnp.float32),
                    jnp.asarray([new_state  ], dtype=jnp.float32),
                    jnp.asarray([float(done)], dtype=jnp.float32),
                )
            )[0])

            per_memory.add(temporal_difference, (state, action, reward, new_state, float(done)))

            if per_memory.tree.n_entries > batch_size:
                per_memory.update_beta(global_steps)

                idxs, batch, is_weights = per_memory.sample(batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

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
                    ),
                    jnp.asarray(is_weights, dtype=jnp.float32)
                )

                new_td_errors_np = np.array(new_td_errors)
                for i in range(batch_size):
                    per_memory.update(idxs[i], float(new_td_errors_np[i]))

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_q_network_params = q_network_params
                if debug:
                    print("сайжруулсан жингүүдийг target неорон сүлжээрүү хууллаа")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
