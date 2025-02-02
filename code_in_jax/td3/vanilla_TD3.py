#
#  TD3 aka Twin Delayed Deep Deterministic Policy Gradient
#

import os
import random
import math
from collections import deque
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import time


debug_render  = True
debug         = False
num_episodes  = 500
learning_rate = 0.0005
gamma         = 0.99
tau           = 0.005
buffer_size   = 10000
batch_size    = 64
policy_noise  = 0.2
noise_clip    = 0.5
policy_delay  = 2


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
    beta_start = 0.4
    beta_frames = 100000
    beta = beta_start
    epsilon = 1e-8
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.experience_count = 0
    def _get_priority(self, error):
        return (error + self.e) ** self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.experience_count = min(self.experience_count + 1, self.tree.capacity)
    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
        priorities = np.array([self.tree.tree[idx] for idx in idxs])
        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.experience_count * sampling_probabilities + self.epsilon, -self.beta)
        is_weights /= is_weights.max() + self.epsilon
        return idxs, batch, is_weights
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    def update_beta(self, frame_idx):
        fraction = min(float(frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)


class Actor(nn.Module):
    action_dim: int
    max_action: float
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=400)(x)
        x = nn.relu(x)
        x = nn.Dense(features=300)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(x)
        x = nn.tanh(x) * self.max_action
        return x

class Critic(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=1)
        x = nn.Dense(features=400)(x)
        x = nn.relu(x)
        x = nn.Dense(features=300)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


env        = gym.make("Pendulum-v1", render_mode='human' if debug_render else None)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor           = Actor(action_dim, max_action)
actor_target    = Actor(action_dim, max_action)
critic_1        = Critic()
critic_1_target = Critic()
critic_2        = Critic()
critic_2_target = Critic()


key = jax.random.PRNGKey(0)
key, actor_key, actor_target_key, critic_1_key, critic_1_target_key, critic_2_key, critic_2_target_key = jax.random.split(key, 7)
actor_params           = actor.init(actor_key, jnp.zeros((1, state_dim)))["params"]
actor_target_params    = actor_target.init(actor_target_key, jnp.zeros((1, state_dim)))["params"]
critic_1_params        = critic_1.init(critic_1_key, jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))["params"]
critic_1_target_params = critic_1_target.init(critic_1_target_key, jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))["params"]
critic_2_params        = critic_2.init(critic_2_key, jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))["params"]
critic_2_target_params = critic_2_target.init(critic_2_target_key, jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))["params"]

actor_optimizer    = optax.adam(learning_rate)
actor_opt_state    = actor_optimizer.init(actor_params)
critic_1_optimizer = optax.adam(learning_rate)
critic_1_opt_state = critic_1_optimizer.init(critic_1_params)
critic_2_optimizer = optax.adam(learning_rate)
critic_2_opt_state = critic_2_optimizer.init(critic_2_params)

per_memory = PERMemory(buffer_size)


@jax.jit
def select_action(actor_params, state):
    return actor.apply({"params": actor_params}, state)

@jax.jit
def critic1_apply(params, state, action):
    return critic_1.apply({"params": params}, state, action)

@jax.jit
def critic2_apply(params, state, action):
    return critic_2.apply({"params": params}, state, action)

@jax.jit
def calculate_td_error(critic_1_params, critic_2_params, actor_target_params, critic_1_target_params, critic_2_target_params, state, action, reward, next_state, done):
    noise       = jnp.clip(jnp.array(np.random.normal(0, policy_noise, size=action.shape), dtype=jnp.float32), -noise_clip, noise_clip)
    next_action = jnp.clip(actor.apply({"params": actor_target_params}, next_state) + noise, -max_action, max_action)
    target_q1   = critic_1_target.apply({"params": critic_1_target_params}, next_state, next_action)
    target_q2   = critic_2_target.apply({"params": critic_2_target_params}, next_state, next_action)
    target_q    = jnp.minimum(target_q1, target_q2)
    td_target   = reward + (1 - done) * gamma * target_q
    current_q1  = critic_1.apply({"params": critic_1_params}, state, action)
    td_error    = td_target - current_q1
    return jnp.abs(td_error)

@jax.jit
def update_critics(critic_1_params, critic_1_opt_state, critic_2_params, critic_2_opt_state, critic_1_target_params, critic_2_target_params, actor_target_params, state, action, reward, next_state, done, is_weights):
    noise       = jnp.clip(jnp.array(np.random.normal(0, policy_noise, size=action.shape), dtype=jnp.float32), -noise_clip, noise_clip)
    next_action = jnp.clip(actor.apply({"params": actor_target_params}, next_state) + noise, -max_action, max_action)
    target_q1   = critic_1_target.apply({"params": critic_1_target_params}, next_state, next_action)
    target_q2   = critic_2_target.apply({"params": critic_2_target_params}, next_state, next_action)
    target_q    = jnp.minimum(target_q1, target_q2)
    target_q    = reward + (1 - done) * gamma * target_q
    def critic_loss_fn(params, state, action, target_q, weights):
        current_q = critic_1.apply({"params": params}, state, action)
        loss      = jnp.mean(weights * (current_q - target_q)**2)
        return loss
    loss1, grads1 = jax.value_and_grad(critic_loss_fn)(critic_1_params, state, action, target_q, is_weights)
    updates1, critic_1_opt_state = critic_1_optimizer.update(grads1, critic_1_opt_state, critic_1_params)
    critic_1_params = optax.apply_updates(critic_1_params, updates1)
    loss2, grads2   = jax.value_and_grad(critic_loss_fn)(critic_2_params, state, action, target_q, is_weights)
    updates2, critic_2_opt_state = critic_2_optimizer.update(grads2, critic_2_opt_state, critic_2_params)
    critic_2_params = optax.apply_updates(critic_2_params, updates2)
    current_q1 = critic_1.apply({"params": critic_1_params}, state, action)
    td_errors  = jnp.abs(target_q - current_q1)
    return critic_1_params, critic_1_opt_state, critic_2_params, critic_2_opt_state, loss1 + loss2, td_errors

@jax.jit
def update_actor(actor_params, actor_opt_state, critic_1_params, state):
    def actor_loss_fn(params, state):
        actions = actor.apply({"params": params}, state)
        loss = -jnp.mean(critic_1.apply({"params": critic_1_params}, state, actions))
        return loss
    loss, grads = jax.value_and_grad(actor_loss_fn)(actor_params, state)
    updates, actor_opt_state = actor_optimizer.update(grads, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    return actor_params, actor_opt_state, loss

@jax.jit
def soft_update(target_params, params, tau):
    return jax.tree_util.tree_map(lambda tp, p: tau * p + (1 - tau) * tp, target_params, params)


global_steps = 0
start_time   = time.time()

for episode in range(num_episodes):
    state, info = env.reset()
    state = np.array(state, dtype=np.float32)
    episode_reward = 0
    while True:
        global_steps += 1
        action = select_action(actor_params, jnp.array([state]))
        action = np.array(action[0], dtype=np.float32)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        init_td_error = calculate_td_error(
            critic_1_params, 
            critic_2_params, 
            actor_target_params, 
            critic_1_target_params, 
            critic_2_target_params, 
            jnp.array([state      ]), 
            jnp.array([action     ]), 
            jnp.array([reward     ]), 
            jnp.array([next_state ]), 
            jnp.array([float(done)]))
        per_memory.add(init_td_error.item(), (state, action, reward, next_state, float(done)))
        state = next_state
        episode_reward += reward
        if per_memory.experience_count > batch_size:
            per_memory.update_beta(global_steps)
            idxs, batch, is_weights = per_memory.sample(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))
            critic_1_params, critic_1_opt_state, critic_2_params, critic_2_opt_state, critic_loss, updated_td_errors = update_critics(
                critic_1_params, 
                critic_1_opt_state, 
                critic_2_params, 
                critic_2_opt_state, 
                critic_1_target_params, 
                critic_2_target_params, 
                actor_target_params,
                jnp.array(state_batch     , dtype=jnp.float32), 
                jnp.array(action_batch    , dtype=jnp.float32), 
                jnp.array(reward_batch    , dtype=jnp.float32),
                jnp.array(next_state_batch, dtype=jnp.float32), 
                jnp.array(done_batch      , dtype=jnp.float32), 
                jnp.array(is_weights      , dtype=jnp.float32)
            )
            for i in range(batch_size):
                idx = idxs[i]
                scalar_td_error = jnp.mean(updated_td_errors[i]).item()
                per_memory.update(idx, scalar_td_error)
            if global_steps % policy_delay == 0:
                actor_params, actor_opt_state, actor_loss = update_actor(
                    actor_params, actor_opt_state, critic_1_params, jnp.array(state_batch, dtype=jnp.float32)
                )
                actor_target_params    = soft_update(actor_target_params, actor_params, tau)
                critic_1_target_params = soft_update(critic_1_target_params, critic_1_params, tau)
                critic_2_target_params = soft_update(critic_2_target_params, critic_2_params, tau)
        if debug_render:
            env.render()
        if done:
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Beta: {per_memory.beta:.3f}")
            break

env.close()
