#
# FQF aka Fully Parameterized Quantile Function + Double DQN + PER 
#

import os
import random
import math
import gymnasium as gym
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
learning_rate = 0.0005
sync_steps    = 50
memory_length = 4000

epsilon       = 1.0
epsilon_decay = 0.00005
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99
hidden_size   = 128

n_quantiles   = 32
n_tau_samples = 64
kappa         = 1.0
tau_clip      = 0.98


class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2*capacity - 1, dtype=np.float32)
        self.data      = np.zeros(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left  = 2*idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
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
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
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
    beta_frames = 200000
    epsilon     = 1e-8

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.beta = self.beta_start
        self.max_priority = 1.0

    def _get_priority(self, error):
        return float((error + self.e) ** self.a)

    def add(self, error, sample):
        p = self._get_priority(error)
        self.max_priority = max(self.max_priority, p)
        self.tree.add(p, sample)

    def add_max(self, sample):
        self.tree.add(self.max_priority, sample)

    def sample(self, n):
        batch   = []
        idxs    = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)

        priorities = np.array([self.tree.tree[idx] for idx in idxs], dtype=np.float32)
        total      = max(self.tree.total(), self.epsilon)
        probs      = priorities / total

        is_weights = np.power(self.tree.n_entries * probs + self.epsilon, -self.beta)
        is_weights /= (is_weights.max() + self.epsilon)

        return idxs, batch, is_weights.astype(np.float32)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.max_priority = max(self.max_priority, p)
        self.tree.update(idx, p)

    def update_beta(self, frame_idx):
        fraction = min(float(frame_idx) / float(self.beta_frames), 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)


class FQFNetwork(nn.Module):
    n_actions    : int
    n_quantiles  : int
    n_tau_samples: int

    def _state_embed(self, x):
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=hidden_size // 2)(x)
        x = nn.relu(x)
        return x

    def _cosine_embed(self, taus):
        i_pi = jnp.arange(1, self.n_tau_samples + 1, dtype=jnp.float32) * jnp.pi
        return jnp.cos(jnp.expand_dims(taus, axis=-1) * i_pi)

    def _quantile_values(self, state_embedding, taus):
        tau_embedding = self._cosine_embed(taus)
        tau_embedding = nn.Dense(features=state_embedding.shape[-1])(tau_embedding)
        tau_embedding = nn.relu(tau_embedding)
        x = tau_embedding * jnp.expand_dims(state_embedding, axis=1)
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x

    @nn.compact
    def __call__(self, x):
        state_embedding = self._state_embed(x)

        fraction_logits = nn.Dense(
            features=self.n_quantiles,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )(state_embedding)

        probs = nn.softmax(fraction_logits, axis=-1)
        probs = jnp.minimum(probs, tau_clip)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)

        taus = jnp.cumsum(probs, axis=-1)
        taus = jnp.concatenate([jnp.zeros((taus.shape[0], 1), dtype=taus.dtype), taus], axis=-1)

        tau_hats = (taus[:, :-1] + taus[:, 1:]) / 2.0

        quantile_values_hats = self._quantile_values(state_embedding, tau_hats)
        quantile_values_tau  = self._quantile_values(state_embedding, taus[:, 1:-1]) if self.n_quantiles > 1 else jnp.zeros((x.shape[0], 0, self.n_actions), dtype=quantile_values_hats.dtype)

        return quantile_values_hats, quantile_values_tau, taus, tau_hats, fraction_logits


env         = gym.make("CartPole-v1", render_mode="human" if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n

fqf_module              = FQFNetwork(n_actions=n_actions, n_quantiles=n_quantiles, n_tau_samples=n_tau_samples)
dummy_input             = jnp.zeros(state.shape, dtype=jnp.float32)
params                  = fqf_module.init(jax.random.PRNGKey(0), jnp.expand_dims(dummy_input, axis=0))
q_network_params        = params["params"]
target_q_network_params = params["params"]

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(q_network_params)

per_memory = PERMemory(memory_length)


@jax.jit
def policy(q_network_params, x):
    x = jnp.expand_dims(x, axis=0)
    quantile_values_hats, _, _, _, _ = fqf_module.apply({"params": q_network_params}, x)
    expected_q_values = jnp.mean(quantile_values_hats, axis=1)
    max_q_action = jnp.argmax(expected_q_values, axis=-1)
    return max_q_action[0], expected_q_values[0]


def _quantile_huber(td_errors):
    abs_td = jnp.abs(td_errors)
    quadratic = jnp.minimum(abs_td, kappa)
    linear = abs_td - quadratic
    huber = 0.5 * quadratic**2 + kappa * linear
    return huber


@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch):
    states      = batch[0]
    actions     = batch[1]
    rewards     = batch[2]
    next_states = batch[3]
    dones       = batch[4]
    is_weights  = batch[5]

    def loss_fn(params):
        quantile_hats, quantile_tau, taus, tau_hats, _ = fqf_module.apply({"params": params}, states)

        next_quantile_hats_online, _, _, _, _ = fqf_module.apply({"params": params}, next_states)
        next_expected_q_values = jnp.mean(next_quantile_hats_online, axis=1)
        next_actions           = jnp.argmax(next_expected_q_values, axis=-1)

        next_quantile_hats_target, _, _, _, _ = fqf_module.apply({"params": target_q_network_params}, next_states)
        next_quantiles    = jnp.take_along_axis(next_quantile_hats_target, next_actions[:, None, None], axis=2).squeeze(axis=2)
        target_quantiles  = rewards[:, None] + gamma * (1.0 - dones[:, None]) * next_quantiles
        current_quantiles = jnp.take_along_axis(quantile_hats, actions[:, None, None], axis=2).squeeze(axis=2)
        td_errors         = target_quantiles[:, None, :] - current_quantiles[:, :, None]

        huber     = _quantile_huber(td_errors)
        indicator = (td_errors < 0.0).astype(jnp.float32)
        weight    = jnp.abs(tau_hats[:, :, None] - indicator)

        quantile_loss_per_sample = jnp.sum(weight * huber / kappa, axis=(1, 2)) / float(n_quantiles)
        quantile_loss = jnp.mean(is_weights * quantile_loss_per_sample)

        if n_quantiles > 1:
            tau_internal         = taus[:, 1:-1]
            q_tau                = jnp.take_along_axis(quantile_tau, actions[:, None, None], axis=2).squeeze(axis=2)
            q_hat                = current_quantiles
            q_hat_left           = q_hat[:, :-1]
            q_hat_right          = q_hat[:, 1: ]
            g                    = 2.0 * q_tau - q_hat_right - q_hat_left
            frac_loss_per_sample = jnp.sum(tau_internal * jax.lax.stop_gradient(g), axis=1)
            frac_loss            = jnp.mean(is_weights * frac_loss_per_sample)
        else:
            frac_loss = 0.0

        loss           = quantile_loss + frac_loss
        new_priorities = jnp.mean(jnp.abs(td_errors), axis=(1, 2))

        return loss, new_priorities

    (loss, new_priorities), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_network_params)
    updates, opt_state = optimizer.update(grads, opt_state, q_network_params)
    q_network_params   = optax.apply_updates(q_network_params, updates)
    return q_network_params, opt_state, loss, new_priorities


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)

        while True:
            global_steps += 1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(q_network_params, state)
                action = int(action)
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            per_memory.add_max((state, action, reward, new_state, float(done)))

            if per_memory.tree.n_entries > batch_size:
                per_memory.update_beta(global_steps)
                idxs, batch, is_weights = per_memory.sample(batch_size)

                state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

                q_network_params, opt_state, loss, new_priorities = train_step(
                    q_network_params,
                    target_q_network_params,
                    opt_state,
                    (
                        jnp.asarray(state_batch     , dtype=jnp.float32),
                        jnp.asarray(action_batch    , dtype=jnp.int32  ),
                        jnp.asarray(reward_batch    , dtype=jnp.float32),
                        jnp.asarray(next_state_batch, dtype=jnp.float32),
                        jnp.asarray(done_batch      , dtype=jnp.float32),
                        jnp.asarray(is_weights      , dtype=jnp.float32),
                    )
                )

                new_priorities_np = np.array(new_priorities, dtype=np.float32)
                for i in range(batch_size):
                    per_memory.update(idxs[i], float(new_priorities_np[i]))

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_q_network_params = q_network_params

            if debug_render:
                env.render()

            if done:
                print("{} episode, reward : {}, epsilon : {:.3f}, beta : {:.3f}".format(episode, sum(episode_rewards), epsilon, per_memory.beta))
                break

finally:
    env.close()
