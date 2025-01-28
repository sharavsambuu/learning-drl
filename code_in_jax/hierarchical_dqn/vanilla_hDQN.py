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

debug_render             = True
num_episodes             = 500
batch_size               = 128         # Increased batch size
learning_rate            = 0.0025      # Increased learning rate
sync_steps               = 250         # Increased sync steps
memory_length            = 4000
meta_controller_interval = 10
epsilon                  = 1.0
epsilon_decay            = 0.0005      # Decreased epsilon decay
epsilon_max              = 1.0
epsilon_min              = 0.01
gamma                    = 0.99

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2*capacity - 1)
        self.data     = np.zeros(capacity, dtype=object)
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
        return self.tree[0]
    def add(self, p, data):
        idx                   = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write           += 1
        if self.write >= self.capacity:
            self.write = 0
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
    a = 0.7             # Increased PER alpha
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    def _get_priority(self, error):
        return (error+self.e)**self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
    def sample(self, n):
        batch   = []
        segment = self.tree.total()/n
        for i in range(n):
            a = segment*i
            b = segment*(i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class MetaControllerNetwork(nn.Module):
    n_meta_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_meta_actions)(x)
        return x

class ControllerNetwork(nn.Module):
    n_actions: int
    n_meta_actions: int

    @nn.compact
    def __call__(self, x, meta_action):
        # x = jnp.expand_dims(x, axis=0) # Remove this
        meta_action_one_hot = jax.nn.one_hot(meta_action, self.n_meta_actions)
        # meta_action_one_hot = jnp.expand_dims(meta_action_one_hot, axis=0)
        combined_input = jnp.concatenate([x, meta_action_one_hot], axis=-1)
        x = nn.Dense(features=64)(combined_input)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x # Remove squeeze as well

env = gym.make('CartPole-v1', render_mode="human")
state, info = env.reset()
n_actions        = env.action_space.n
n_meta_actions   = n_actions

meta_controller_module = MetaControllerNetwork(n_meta_actions=n_meta_actions)
meta_controller_params = meta_controller_module.init(jax.random.PRNGKey(0), jnp.asarray(state))['params']
target_meta_controller_params = meta_controller_params

controller_module = ControllerNetwork(n_actions=n_actions, n_meta_actions=n_meta_actions)
controller_params = controller_module.init(jax.random.PRNGKey(0), jnp.asarray(state), jnp.asarray(0, dtype=jnp.int32))['params']
target_controller_params = controller_params

meta_controller_optimizer = optax.adam(learning_rate)
meta_controller_opt_state = meta_controller_optimizer.init(meta_controller_params)

controller_optimizer = optax.adam(learning_rate)
controller_opt_state = controller_optimizer.init(controller_params)

per_memory = PERMemory(memory_length)

@jax.jit
def meta_controller_policy(params, x):
    predicted_q_values = meta_controller_module.apply({'params': params}, x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values

@jax.jit
def controller_policy(params, x, meta_action):
    predicted_q_values = controller_module.apply({'params': params}, x, meta_action)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values

def calculate_td_error(q_value_vec, target_q_value_vec, action, reward, n_actions):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value         = jnp.sum(one_hot_actions*q_value_vec, axis=-1)
    td_target       = reward + gamma*jnp.max(target_q_value_vec, axis=-1)
    td_error        = td_target - q_value
    return jnp.abs(td_error)

def td_error_meta_controller(meta_controller_params, target_meta_controller_params, batch):
    states, meta_actions, rewards, next_states, dones = batch
    predicted_q_values = meta_controller_module.apply({'params': meta_controller_params}, states)
    target_q_values    = meta_controller_module.apply({'params': target_meta_controller_params}, next_states)
    return calculate_td_error(predicted_q_values, target_q_values, meta_actions, rewards, n_meta_actions)

def td_error_controller(controller_params, target_controller_params, batch, meta_action):
    states, actions, rewards, next_states, dones = batch
    predicted_q_values = controller_module.apply({'params': controller_params}, states, meta_action)
    target_q_values    = controller_module.apply({'params': target_controller_params}, next_states, meta_action)
    return calculate_td_error(predicted_q_values, target_q_values, actions, rewards, n_actions)

def q_learning_loss(q_value_vec, target_q_value_vec, action, reward, done):
    one_hot_actions = jax.nn.one_hot(action, q_value_vec.shape[-1])
    q_value = jnp.sum(q_value_vec * one_hot_actions, axis=-1)
    td_target = reward + gamma * jnp.max(target_q_value_vec, axis=-1) * (1.0 - done)
    td_error = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

@jax.jit
def train_step_meta_controller(meta_controller_params, target_meta_controller_params, meta_controller_opt_state, batch):
    def loss_fn(params):
        predicted_q_values = meta_controller_module.apply({'params': params}, batch[0])
        target_q_values    = meta_controller_module.apply({'params': target_meta_controller_params}, batch[3])
        return jnp.mean(
            q_learning_loss(
                predicted_q_values,
                target_q_values,
                batch[1],
                batch[2],
                batch[4]
            )
        )
    loss, gradients = jax.value_and_grad(loss_fn)(meta_controller_params)
    updates, meta_controller_opt_state = meta_controller_optimizer.update(gradients, meta_controller_opt_state, meta_controller_params)
    meta_controller_params = optax.apply_updates(meta_controller_params, updates)
    return meta_controller_params, meta_controller_opt_state, loss, td_error_meta_controller(meta_controller_params, target_meta_controller_params, batch)

@jax.jit
def train_step_controller(controller_params, target_controller_params, controller_opt_state, batch, meta_action):
    # meta_actions = jnp.repeat(jnp.array([meta_action]), batch_size) # Removed this line
    def loss_fn(params):
        predicted_q_values = controller_module.apply({'params': params}, batch[0], batch[1]) # Use batched meta_actions
        target_q_values    = controller_module.apply({'params': target_controller_params}, batch[3], batch[1]) # Use batched meta_actions
        return jnp.mean(
            q_learning_loss(
                predicted_q_values,
                target_q_values,
                batch[1],
                batch[2],
                batch[4]
            )
        )
    loss, gradients = jax.value_and_grad(loss_fn)(controller_params)
    updates, controller_opt_state = controller_optimizer.update(gradients, controller_opt_state, controller_params)
    controller_params = optax.apply_updates(controller_params, updates)
    return controller_params, controller_opt_state, loss, td_error_controller(controller_params, target_controller_params, batch, batch[1]) # Use batched meta_actions

global_steps = 0
meta_action = None
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        meta_controller_steps = 0
        while True:
            global_steps += 1
            meta_controller_steps += 1

            if meta_controller_steps % meta_controller_interval == 1:
                if np.random.rand() <= epsilon:
                    meta_action = env.action_space.sample()
                else:
                    meta_action, _ = meta_controller_policy(meta_controller_params, jnp.asarray(state))
                    meta_action = int(meta_action)

                meta_controller_steps = 1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = controller_policy(controller_params, jnp.asarray(state), jnp.asarray(meta_action, dtype=jnp.int32))
                action = int(action)

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            temporal_difference = per_memory.tree.total()
            per_memory.add(temporal_difference, (state, meta_action, reward, new_state, float(done), action))

            if len(per_memory.tree.data) > batch_size:
                batch = per_memory.sample(batch_size)
                states, meta_actions_batch, rewards, next_states, dones, actions = [], [], [], [], [], []
                for i in range(batch_size):
                    states.append(batch[i][1][0])
                    meta_actions_batch.append(batch[i][1][1])
                    rewards.append(batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones.append(batch[i][1][4])
                    actions.append(batch[i][1][5])
                meta_controller_params, meta_controller_opt_state, meta_controller_loss, _ = train_step_meta_controller(
                    meta_controller_params, target_meta_controller_params, meta_controller_opt_state,
                    (jnp.asarray(states), jnp.asarray(meta_actions_batch, dtype=jnp.int32), jnp.asarray(rewards, dtype=jnp.float32), jnp.asarray(next_states), jnp.asarray(dones, dtype=jnp.float32))
                )
                controller_params, controller_opt_state, controller_loss, new_td_errors = train_step_controller(
                    controller_params, target_controller_params, controller_opt_state,
                    (jnp.asarray(states), jnp.asarray(meta_actions_batch, dtype=jnp.int32), jnp.asarray(rewards, dtype=jnp.float32), jnp.asarray(next_states), jnp.asarray(dones, dtype=jnp.float32)), # Use meta_actions_batch here as well
                    jnp.asarray(meta_actions_batch, dtype=jnp.int32)
                )

                new_td_errors = np.array(new_td_errors)
                for i in range(batch_size):
                    idx = batch[i][0]
                    per_memory.update(idx, new_td_errors[i])

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_meta_controller_params = meta_controller_params
                target_controller_params = controller_params

            if debug_render:
                env.render()

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break
finally:
    env.close()