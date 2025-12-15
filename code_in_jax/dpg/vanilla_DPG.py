#
# DPG aka Deterministic Policy Gradient, one of precursors of TD3
#
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

debug_render     = True
num_episodes     = 500
learning_rate    = 0.0005
gamma            = 0.99
tau              = 0.005
buffer_size      = 10000
batch_size       = 64

# exploration
action_noise_std = 0.10   # 0.05 ~ 0.30 for Pendulum


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1)
        self.data     = np.zeros(capacity, dtype=object)
        self.write    = 0  # instance variable (not class variable)

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
        change        = p - self.tree[idx]
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
    beta        = beta_start
    epsilon     = 1e-8

    def __init__(self, capacity):
        self.tree             = SumTree(capacity)
        self.experience_count = 0

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.experience_count = min(self.experience_count + 1, self.tree.capacity)

    def sample(self, n):
        batch   = []
        idxs    = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)

        priorities             = np.array([self.tree.tree[idx] for idx in idxs], dtype=np.float32)
        sampling_probabilities = priorities / (self.tree.total() + self.epsilon)
        is_weights             = np.power(self.experience_count * sampling_probabilities + self.epsilon, -self.beta)
        is_weights            /= (is_weights.max() + self.epsilon)

        return idxs, batch, is_weights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def update_beta(self, frame_idx):
        fraction  = min(float(frame_idx) / self.beta_frames, 1.0)
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


actor         = Actor(action_dim, max_action)
actor_target  = Actor(action_dim, max_action)
critic        = Critic()
critic_target = Critic()

key = jax.random.PRNGKey(0)
key, actor_key, actor_target_key, critic_key, critic_target_key = jax.random.split(key, 5)

actor_params         = actor        .init(actor_key        , jnp.zeros((1, state_dim))                            )["params"]
actor_target_params  = actor_target .init(actor_target_key , jnp.zeros((1, state_dim))                            )["params"]
critic_params        = critic       .init(critic_key       , jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))["params"]
critic_target_params = critic_target.init(critic_target_key, jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))["params"]

# start target nets as exact copies (standard DDPG/DPG behavior)
actor_target_params  = actor_params
critic_target_params = critic_params

actor_optimizer  = optax.adam(learning_rate)
actor_opt_state  = actor_optimizer.init(actor_params)
critic_optimizer = optax.adam(learning_rate)
critic_opt_state = critic_optimizer.init(critic_params)

per_memory       = PERMemory(buffer_size)


@jax.jit
def select_action(actor_params, state):
    return actor.apply({"params": actor_params}, state)


@jax.jit
def calculate_td_error(critic_params, critic_target_params, actor_target_params, state, action, reward, next_state, done):
    # Make shapes consistent: (B,1)
    reward = reward.reshape(-1, 1)
    done   = done  .reshape(-1, 1)

    next_action = actor        .apply({"params": actor_target_params }, next_state             )
    target_q    = critic_target.apply({"params": critic_target_params}, next_state, next_action)
    td_target   = reward + (1.0 - done) * gamma * target_q

    current_q = critic.apply({"params": critic_params}, state, action)
    td_error  = td_target - current_q

    return jnp.abs(td_error)


@jax.jit
def update_critic(critic_params, critic_opt_state, critic_target_params, actor_target_params,
                  state, action, reward, next_state, done, is_weights):

    # Make shapes consistent: (B,1)
    reward     = reward    .reshape(-1, 1)
    done       = done      .reshape(-1, 1)
    is_weights = is_weights.reshape(-1, 1)

    next_action = actor        .apply({"params": actor_target_params }, next_state             )
    target_q    = critic_target.apply({"params": critic_target_params}, next_state, next_action)

    # stop_gradient is a safety belt, targets should not backprop anyway
    target_q  = jax.lax.stop_gradient(target_q)
    target_q  = reward + (1.0 - done) * gamma * target_q

    def critic_loss_fn(params, state, action, target_q, weights):
        current_q = critic.apply({"params": params}, state, action)
        loss      = jnp.mean(weights * (current_q - target_q) ** 2)
        return loss

    loss, grads = jax.value_and_grad(critic_loss_fn)(critic_params, state, action, target_q, is_weights)
    updates, critic_opt_state = critic_optimizer.update(grads, critic_opt_state, critic_params)
    critic_params             = optax.apply_updates(critic_params, updates)

    current_q  = critic.apply({"params": critic_params}, state, action)
    td_errors  = jnp.abs(target_q - current_q)

    return critic_params, critic_opt_state, loss, td_errors


@jax.jit
def update_actor(actor_params, actor_opt_state, critic_params, state):
    def actor_loss_fn(params, state):
        actions = actor.apply({"params": params}, state)
        loss    = -jnp.mean(critic.apply({"params": critic_params}, state, actions))
        return loss

    loss, grads              = jax.value_and_grad(actor_loss_fn)(actor_params, state)
    updates, actor_opt_state = actor_optimizer.update(grads, actor_opt_state, actor_params)
    actor_params             = optax.apply_updates(actor_params, updates)

    return actor_params, actor_opt_state, loss


@jax.jit
def soft_update(target_params, params, tau):
    return jax.tree_util.tree_map(lambda tp, p: tau * p + (1.0 - tau) * tp, target_params, params)


global_steps = 0
start_time   = time.time()

for episode in range(num_episodes):
    state, info    = env.reset()
    state          = np.array(state, dtype=np.float32)
    episode_reward = 0.0

    while True:
        global_steps += 1

        # deterministic action from actor
        action = select_action(actor_params, jnp.array([state], dtype=jnp.float32))
        action = np.array(action[0], dtype=np.float32)

        # exploration noise 
        action = action + np.random.normal(0.0, action_noise_std, size=action.shape).astype(np.float32)
        action = np.clip(action, -max_action, max_action)

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done       = terminated or truncated

        # TD error for PER priority
        td_error = calculate_td_error(
            critic_params       ,
            critic_target_params,
            actor_target_params ,
            jnp.array([state      ], dtype=jnp.float32),
            jnp.array([action     ], dtype=jnp.float32),
            jnp.array([reward     ], dtype=jnp.float32),
            jnp.array([next_state ], dtype=jnp.float32),
            jnp.array([float(done)], dtype=jnp.float32)
        )

        per_memory.add(td_error.item(), (state, action, reward, next_state, float(done)))

        state           = next_state
        episode_reward += reward

        if per_memory.experience_count > batch_size:
            per_memory.update_beta(global_steps)

            idxs, batch, is_weights = per_memory.sample(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

            critic_params, critic_opt_state, critic_loss, updated_td_errors = update_critic(
                critic_params,
                critic_opt_state,
                critic_target_params,
                actor_target_params,
                jnp.array(state_batch,      dtype=jnp.float32),
                jnp.array(action_batch,     dtype=jnp.float32),
                jnp.array(reward_batch,     dtype=jnp.float32),
                jnp.array(next_state_batch, dtype=jnp.float32),
                jnp.array(done_batch,       dtype=jnp.float32),
                jnp.array(is_weights,       dtype=jnp.float32)
            )

            # update PER priorities with latest td errors
            for i in range(batch_size):
                idx = idxs[i]
                per_memory.update(idx, updated_td_errors[i].item())

            actor_params, actor_opt_state, actor_loss = update_actor(
                actor_params,
                actor_opt_state,
                critic_params,
                jnp.array(state_batch, dtype=jnp.float32)
            )

            actor_target_params  = soft_update(actor_target_params,  actor_params,  tau)
            critic_target_params = soft_update(critic_target_params, critic_params, tau)

        if debug_render:
            env.render()

        if done:
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Beta: {per_memory.beta:.3f}")
            break

env.close()
