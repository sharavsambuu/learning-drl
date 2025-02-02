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


debug_render = True
num_episodes = 500
gamma        = 0.99
tau          = 0.005
buffer_size  = 10000
batch_size   = 64
policy_noise = 0.2
noise_clip   = 0.5
policy_freq  = 2
actor_lr     = 0.0005
critic_lr    = 0.0005


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
    def _get_priority(self, error):
        return (error + self.e) ** self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
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
        return idxs, batch
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

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
        x = jnp.concatenate([state, action], axis=-1)
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
key, actor_key, actor_target_key, critic_1_key, critic_1_target_key, critic_2_key, critic_2_target_key = jax.random.split(
    key, 7)

dummy_state = jnp.zeros((1, state_dim))
dummy_action = jnp.zeros((1, action_dim))

actor_params = actor.init(actor_key, dummy_state)["params"]
actor_target_params = actor_target.init(actor_target_key, dummy_state)["params"]
critic_1_params = critic_1.init(critic_1_key, dummy_state, dummy_action)["params"]
critic_1_target_params = critic_1_target.init(critic_1_target_key, dummy_state, dummy_action)["params"]
critic_2_params = critic_2.init(critic_2_key, dummy_state, dummy_action)["params"]
critic_2_target_params = critic_2_target.init(critic_2_target_key, dummy_state, dummy_action)["params"]

actor_optimizer = optax.adam(actor_lr)
actor_opt_state = actor_optimizer.init(actor_params)
critic_1_optimizer = optax.adam(critic_lr)
critic_1_opt_state = critic_1_optimizer.init(critic_1_params)
critic_2_optimizer = optax.adam(critic_lr)
critic_2_opt_state = critic_2_optimizer.init(critic_2_params)

per_memory = PERMemory(buffer_size)

@jax.jit
def select_action(actor_params, state):
    return actor.apply({"params": actor_params}, state)

@jax.jit
def target_policy_smoothing(actor_target_params, next_state, key):
    noise = jnp.clip(jax.random.normal(key, (1, action_dim)) * policy_noise, -noise_clip, noise_clip) # Concrete shape
    next_action = actor_target.apply({"params": actor_target_params}, next_state) + noise
    return jnp.clip(next_action, -max_action, max_action)

@jax.jit
def calculate_td_error(actor_target_params, critic_1_params, critic_2_params, state, action, reward, next_state, done, key):
    key, noise_key = jax.random.split(key)
    next_action = target_policy_smoothing(actor_target_params, next_state, noise_key)
    target_q1 = critic_1_target.apply({"params": critic_1_target_params}, next_state, next_action)
    target_q2 = critic_2_target.apply({"params": critic_2_target_params}, next_state, next_action)
    target_q = jnp.minimum(target_q1, target_q2)
    target_q = reward + (1 - done) * gamma * target_q
    current_q1 = critic_1.apply({"params": critic_1_params}, state, action)
    current_q2 = critic_2.apply({"params": critic_2_params}, state, action)
    td_error = jnp.abs(target_q - current_q1) + jnp.abs(
        target_q - current_q2)
    return jnp.mean(td_error)

@jax.jit
def update_critic(critic_1_params, critic_1_opt_state, critic_2_params, critic_2_opt_state, actor_target_params,
                  state, action, reward, next_state, done, key):
    key, noise_key = jax.random.split(key)
    next_action = target_policy_smoothing(actor_target_params, next_state, noise_key)
    target_q1 = critic_1_target.apply({"params": critic_1_target_params}, next_state, next_action)
    target_q2 = critic_2_target.apply({"params": critic_2_target_params}, next_state, next_action)
    target_q = jnp.minimum(target_q1, target_q2)
    target_q = reward + (1 - done) * gamma * target_q

    def critic_1_loss_fn(params):
        current_q = critic_1.apply({"params": params}, state, action)
        loss = jnp.mean((current_q - target_q) ** 2)
        return loss

    loss_1, grads_1 = jax.value_and_grad(critic_1_loss_fn)(critic_1_params)
    updates_1, critic_1_opt_state = critic_1_optimizer.update(grads_1, critic_1_opt_state, critic_1_params)
    critic_1_params = optax.apply_updates(critic_1_params, updates_1)

    def critic_2_loss_fn(params):
        current_q = critic_2.apply({"params": params}, state, action)
        loss = jnp.mean((current_q - target_q) ** 2)
        return loss

    loss_2, grads_2 = jax.value_and_grad(critic_2_loss_fn)(critic_2_params)
    updates_2, critic_2_opt_state = critic_2_optimizer.update(grads_2, critic_2_opt_state, critic_2_params)
    critic_2_params = optax.apply_updates(critic_2_params, updates_2)

    return critic_1_params, critic_1_opt_state, critic_2_params, critic_2_opt_state, loss_1, loss_2, key

@jax.jit
def update_actor(actor_params, actor_opt_state, critic_1_params, state):
    def actor_loss_fn(params):
        actions = actor.apply({"params": params}, state)
        loss = -jnp.mean(critic_1.apply({"params": critic_1_params}, state, actions))
        return loss

    loss, grads = jax.value_and_grad(actor_loss_fn)(actor_params)
    updates, actor_opt_state = actor_optimizer.update(grads, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    return actor_params, actor_opt_state, loss

@jax.jit
def soft_update(target_params, params, tau):
    return jax.tree_util.tree_map(lambda tp, p: tau * p + (1 - tau) * tp, target_params, params)


total_it        = 0
episode_rewards = []
key = jax.random.PRNGKey(0)
for episode in range(num_episodes):
    state, info = env.reset()
    state = np.array(state, dtype=np.float32)
    episode_reward = 0
    while True:
        total_it += 1
        key, action_key, td_error_key, critic_key = jax.random.split(key, 4)
        action = select_action(actor_params, jnp.array([state]))
        action = np.array(action[0], dtype=np.float32)

        action_for_critic = action.reshape(1, -1)

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated
        episode_reward += reward
        td_error = calculate_td_error(
            actor_target_params,
            critic_1_params,
            critic_2_params,
            jnp.array([state]),
            jnp.array(action_for_critic),
            jnp.array([reward]),
            jnp.array([next_state]),
            jnp.array([float(done)]),
            td_error_key
        )
        per_memory.add(td_error.item(), (state, action, reward, next_state, float(done)))

        state = next_state
        if per_memory.tree.write > batch_size:
            idxs, batch = per_memory.sample(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))
            key, critic_update_key, per_update_key, actor_update_key = jax.random.split(key, 4)

            critic_1_params, critic_1_opt_state, critic_2_params, critic_2_opt_state, critic_1_loss, critic_2_loss, key = \
                update_critic(
                    critic_1_params,
                    critic_1_opt_state,
                    critic_2_params,
                    critic_2_opt_state,
                    actor_target_params,
                    jnp.array(state_batch, dtype=jnp.float32),
                    jnp.array(action_batch, dtype=jnp.float32),
                    jnp.array(reward_batch, dtype=jnp.float32),
                    jnp.array(next_state_batch, dtype=jnp.float32),
                    jnp.array(done_batch, dtype=jnp.float32),
                    critic_update_key
                )

            for i in range(batch_size):
                idx = idxs[i]
                td_error = calculate_td_error(
                    actor_target_params,
                    critic_1_params,
                    critic_2_params,
                    jnp.array([state_batch[i]     ]),
                    jnp.array([action_batch[i]    ]),
                    jnp.array([reward_batch[i]    ]),
                    jnp.array([next_state_batch[i]]),
                    jnp.array([done_batch[i]      ]),
                    per_update_key
                )
                per_memory.update(idx, td_error.item())

            if total_it % policy_freq == 0:
                actor_params, actor_opt_state, actor_loss = update_actor(
                    actor_params, actor_opt_state, critic_1_params, jnp.array(state_batch, dtype=jnp.float32)
                )

                actor_target_params = soft_update(actor_target_params, actor_params, tau)
                critic_1_target_params = soft_update(critic_1_target_params, critic_1_params, tau)
                critic_2_target_params = soft_update(critic_2_target_params, critic_2_params, tau)

        if debug_render:
            env.render()

        if done:
            episode_rewards.append(episode_reward)
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
            break

env.close()