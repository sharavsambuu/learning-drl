import os
import random
import math
import gymnasium as gym
from collections import deque

import flax.linen as nn
import jax
from jax import jit, vmap
from jax import numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt


debug_render  = True
num_episodes  = 500
batch_size    = 64
learning_rate = 0.0005
sync_steps    = 50
memory_length = 4000
n_quantiles   = 32
n_tau_samples = 64
epsilon       = 1.0
epsilon_decay = 0.00005
epsilon_max   = 1.0
epsilon_min   = 0.01
gamma         = 0.99
hidden_size   = 128


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
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
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PERMemory:
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity # Added capacity to PERMemory

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            assert isinstance(data, tuple), f"Data is not a tuple, but {type(data)}" # Assertion to check data type
            batch.append((idx, data))
        assert len(batch) == n, f"Batch size is not equal to n, batch_size={len(batch)}, n={n}" # Assertion to check batch size
        return batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)



# FQF Network
class FQFNetwork(nn.Module):
    n_actions: int
    n_quantiles: int
    n_tau_samples: int

    @nn.compact
    def __call__(self, x):
        # State embedding
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=hidden_size // 2)(x)
        state_embedding = nn.relu(x)

        # Fraction proposal network
        fraction_layer = nn.Dense(features=self.n_quantiles)(state_embedding)
        fraction_logits = nn.relu(fraction_layer)
        tau_hats = nn.softmax(fraction_logits, axis=-1)
        tau_hats = jnp.cumsum(tau_hats, axis=-1)

        # generate tau_i
        tau_0 = jnp.zeros((x.shape[0], 1))
        tau_hats = jnp.concatenate([tau_0, tau_hats], axis=-1)

        tau_i = (tau_hats[:, 1:] + tau_hats[:, :-1]) / 2.
        tau_i = jnp.tile(tau_i[:, jnp.newaxis, :], (1, self.n_actions, 1))

        # Cosine embedding of tau_i
        i_pi = jnp.arange(1, self.n_tau_samples + 1, dtype=jnp.float32) * jnp.pi
        tau_i_embedding = jnp.cos(tau_i[..., jnp.newaxis] * i_pi)
        tau_i_embedding = tau_i_embedding.reshape(x.shape[0], self.n_actions, -1)

        # Value network
        # Concatenate state embedding with cosine embedding
        state_embedding_expanded = jnp.tile(state_embedding[:, jnp.newaxis, :], (1, self.n_actions, 1))
        x = jnp.concatenate([state_embedding_expanded, tau_i_embedding], axis=-1)
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        quantile_values = nn.Dense(features=self.n_quantiles)(x)

        return quantile_values, tau_hats


# Create environment
env = gym.make('CartPole-v1', render_mode="human")
state, info = env.reset()

# Instantiate network and optimizer
n_actions   = env.action_space.n
fqf_module  = FQFNetwork(n_actions=n_actions, n_quantiles=n_quantiles, n_tau_samples=n_tau_samples)
dummy_state = jnp.zeros_like(state)
params                  = fqf_module.init(jax.random.PRNGKey(0), jnp.expand_dims(dummy_state, axis=0))
q_network_params        = params['params']
target_q_network_params = params['params']

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(q_network_params)

# PER Memory
per_memory = PERMemory(memory_length)

def policy(params, x):
    x = jnp.expand_dims(x, axis=0)
    predicted_q_values, _ = fqf_module.apply({'params': params}, x)
    expected_q_values = jnp.mean(predicted_q_values, axis=-1)
    max_q_action = jnp.argmax(expected_q_values, axis=-1)
    return max_q_action[0], expected_q_values[0]

@jit
def calculate_quantile_values(params, x, n_quantiles_sample):
    quantile_values, tau_hats = fqf_module.apply({'params': params}, x)
    return quantile_values, tau_hats

# Huber loss
@jit
def huber_loss(td_errors, tau_i):
    delta = 1.0
    abs_td_errors = jnp.abs(td_errors)
    quadratic = jnp.minimum(abs_td_errors, delta)
    linear = abs_td_errors - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    tau_i_expanded = jnp.expand_dims(tau_i, axis=-1)
    quantile_huber_loss = jnp.abs(tau_i_expanded - (td_errors < 0.0).astype(jnp.float32)) * loss
    return jnp.mean(jnp.sum(quantile_huber_loss, axis=1))

@jit
def train_step(q_network_params, target_q_network_params, opt_state, batch, key):
    def loss_fn(params):
        predicted_quantile_values, tau_hats = fqf_module.apply({'params': params}, batch[0])
        target_quantile_values, _ = fqf_module.apply({'params': target_q_network_params}, batch[3])

        tau_i = (tau_hats[:, 1:] + tau_hats[:, :-1]) / 2.
        tau_i_selected = tau_i[jnp.arange(batch_size), batch[1]]

        expected_next_q_values = jnp.mean(target_quantile_values, axis=2)
        max_next_actions = jnp.argmax(expected_next_q_values, axis=1)
        next_state_quantile_values = target_quantile_values[jnp.arange(batch_size), max_next_actions]

        target_quantile_values = batch[2][:, jnp.newaxis] + gamma * (1 - batch[4][:, jnp.newaxis]) * next_state_quantile_values
        predicted_action_quantile_values = predicted_quantile_values[jnp.arange(batch_size), batch[1]]
        td_errors = target_quantile_values - predicted_action_quantile_values
        loss = huber_loss(td_errors, tau_i_selected)
        return loss, td_errors

    (loss, td_errors), gradients = jax.value_and_grad(loss_fn, has_aux=True)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params = optax.apply_updates(q_network_params, updates)
    new_priorities = jnp.abs(jnp.mean(td_errors, axis=1))
    return q_network_params, opt_state, loss, new_priorities

rng = jax.random.PRNGKey(0)
global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        while True:
            global_steps = global_steps + 1
            rng, policy_key, train_key = jax.random.split(rng, 3)

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(q_network_params, state)
                action = int(action)

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            per_memory.add(0, (state, action, reward, new_state, float(done)))

            if len(per_memory.tree.data) >= batch_size:
                batch = per_memory.sample(batch_size)
                states, actions, rewards, next_states, dones = [], [], [], [], []
                for i in range(batch_size):
                    data = batch[i][1]
                    states.append(data[0])
                    actions.append(data[1])
                    rewards.append(data[2])
                    next_states.append(data[3])
                    dones.append(data[4])

                q_network_params, opt_state, loss, new_priorities = train_step(
                    q_network_params,
                    target_q_network_params,
                    opt_state,
                    (
                        jnp.asarray(states),
                        jnp.asarray(actions, dtype=jnp.int32),
                        jnp.asarray(rewards, dtype=jnp.float32),
                        jnp.asarray(next_states),
                        jnp.asarray(dones, dtype=jnp.float32)
                    ),
                    train_key
                )
                for i in range(batch_size):
                    idx = batch[i][0]
                    per_memory.update(idx, new_priorities[i])

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_q_network_params = q_network_params

            if debug_render:
                env.render()

            if done:
                print("{} episode, reward : {}".format(episode, sum(episode_rewards)))
                break

finally:
    env.close()