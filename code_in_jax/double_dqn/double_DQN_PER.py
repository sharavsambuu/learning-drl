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
    a = 0.6
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


class DeepQNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_layer       = nn.Dense(features=self.n_actions)(activation_layer_2)
        return output_layer


env         = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()
state       = np.array(state, dtype=np.float32)

n_actions               = env.action_space.n

dqn_module              = DeepQNetwork(n_actions=n_actions)
dummy_input             = jnp.zeros(state.shape)
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


def calculate_td_error(q_value_vec, next_q_value_vec, target_q_value_vec, action, reward):
    one_hot_actions    = jax.nn.one_hot(action, n_actions)
    q_value            = jnp.sum(one_hot_actions*q_value_vec)
    action_select      = jnp.argmax(next_q_value_vec)
    target_q_value     = target_q_value_vec[action_select]
    td_target          = reward + gamma*target_q_value
    td_error           = td_target - q_value
    return jnp.abs(td_error)

calculate_td_error_vmap = jax.vmap(calculate_td_error, in_axes=(0, 0, 0, 0, 0), out_axes=0)


def q_learning_loss(q_value_vec, next_q_value_vec, target_q_value_vec, action, reward, done):
    one_hot_actions    = jax.nn.one_hot(action, n_actions)
    q_value            = jnp.sum(one_hot_actions*q_value_vec)
    action_select      = jnp.argmax(next_q_value_vec)
    target_q_value     = target_q_value_vec[action_select]
    td_target          = reward + gamma*target_q_value*(1.-done)
    td_error           = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

q_learning_loss_vmap = jax.vmap(q_learning_loss, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)


@jax.jit
def td_error_batch(q_network_params, target_q_network_params, batch):
    predicted_q_values      = dqn_module.apply({'params': q_network_params}, batch[0])
    predicted_next_q_values = dqn_module.apply({'params': q_network_params}, batch[3])
    target_q_values         = dqn_module.apply({'params': target_q_network_params}, batch[3])
    return calculate_td_error_vmap(predicted_q_values, predicted_next_q_values, target_q_values, batch[1], batch[2])


@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch):
    def loss_fn(params):
        predicted_q_values      = dqn_module.apply({'params': params}, batch[0])
        predicted_next_q_values = dqn_module.apply({'params': params}, batch[3])
        target_q_values         = dqn_module.apply({'params': target_q_network_params}, batch[3])
        return jnp.mean(
                q_learning_loss_vmap(
                    predicted_q_values,
                    predicted_next_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients    = jax.value_and_grad(loss_fn)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params   = optax.apply_updates(q_network_params, updates)
    current_td_errors  = td_error_batch(q_network_params, target_q_network_params, batch) # Calculate td_error here and return
    return q_network_params, opt_state, loss, current_td_errors


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(q_network_params, state)

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            temporal_difference = float(td_error_batch(q_network_params, target_q_network_params, (
                    jnp.asarray([state]),
                    jnp.asarray([action]),
                    jnp.asarray([reward]),
                    jnp.asarray([new_state])
                ))[0])
            per_memory.add(temporal_difference, (state, action, reward, new_state, float(done)))

            if (len(per_memory.tree.data)>batch_size):
                batch = per_memory.sample(batch_size)
                idxs, segment_data = zip(*batch)
                states, actions, rewards, next_states, dones = [], [], [], [], []
                for data in segment_data:
                    states.append     (data[0])
                    actions.append    (data[1])
                    rewards.append    (data[2])
                    next_states.append(data[3])
                    dones.append      (data[4])

                q_network_params, opt_state, loss, new_td_errors = train_step(
                                        q_network_params,
                                        target_q_network_params,
                                        opt_state,
                                        (
                                            jnp.asarray(list(states)),
                                            jnp.asarray(list(actions), dtype=jnp.int32),
                                            jnp.asarray(list(rewards), dtype=jnp.float32),
                                            jnp.asarray(list(next_states)),
                                            jnp.asarray(list(dones), dtype=jnp.float32)
                                        )
                                    )
                new_td_errors_np = np.array(new_td_errors)
                for i in range(batch_size):
                    idx = idxs[i]
                    per_memory.update(idx, new_td_errors_np[i])


            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0:
                target_q_network_params = q_network_params # Directly copy params
                if debug:
                    print("сайжруулсан жингүүдийг target неорон сүлжээрүү хууллаа")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()