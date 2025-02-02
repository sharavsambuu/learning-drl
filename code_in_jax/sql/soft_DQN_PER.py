# Soft Q-Learning (SQL) - Updated for Flax Linen and Optax
# References :
#   - https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
#   - https://arxiv.org/pdf/1702.08165.pdf
#   - https://zhuanlan.zhihu.com/p/150527098
#   - https://en.wikipedia.org/wiki/LogSumExp


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
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000
tau           = 0.005   # Temperature parameter for Soft Q-Learning
gamma         = 0.99    # Discount factor


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


env = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()
state = np.array(state, dtype=np.float32)
n_actions = env.action_space.n

# Initialize the Soft Q-Network
dqn_module  = SoftQNetwork(n_actions=n_actions)
dummy_input = jnp.zeros(state.shape)
params      = dqn_module.init(jax.random.PRNGKey(0), dummy_input)
q_network_params        = params['params']
target_q_network_params = params['params']

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(q_network_params)

per_memory = PERMemory(memory_length)


@jax.jit
def policy(params, x):
    q_values = dqn_module.apply({'params': params}, x)
    policy_probabilities = nn.softmax(q_values/tau)
    # Due to some errors like NaN values because of small tau in softmax, I will use log_softmax
    # log_policy_probabilities = nn.log_softmax(q_values/tau)
    return policy_probabilities

def calculate_td_error(q_value_vec, target_q_value, action, reward):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value = jnp.sum(one_hot_actions * q_value_vec)
    td_error = reward + gamma * target_q_value - q_value
    return jnp.abs(td_error)

calculate_td_error_vmap = jax.vmap(calculate_td_error, in_axes=(0, 0, 0, 0), out_axes=0)

@jax.jit
def td_error_batch(q_network_params, target_q_network_params, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    predicted_q_values = dqn_module.apply({'params': q_network_params}, batch[0])
    target_q_values = dqn_module.apply({'params': target_q_network_params}, batch[3])

    # Calculate target based on log-sum-exp
    target_v = jax.scipy.special.logsumexp(target_q_values / tau, axis=1) - jnp.log(n_actions)
    target_q_value = tau * target_v
    # The above 2 lines are used instead of:
    # target_q_values = jax.nn.softmax(target_q_values / tau)
    # target_q_value = jnp.sum(target_q_values * (target_q_values - tau * jnp.log(target_q_values)), axis=1)

    return calculate_td_error_vmap(predicted_q_values, target_q_value, batch[1], batch[2])

def soft_q_learning_loss(q_value_vec, target_q_value, action, reward, done):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value = jnp.sum(one_hot_actions * q_value_vec)
    
    # Calculate target based on log-sum-exp
    # target_v = jax.scipy.special.logsumexp(target_q_value / tau, axis=1) - jnp.log(n_actions)
    # target_q = tau * target_v
    # The above 2 lines are used instead of:
    # target_q_values = jax.nn.softmax(target_q_value / tau)
    # target_q = jnp.sum(target_q_values * (target_q_values - tau * jnp.log(target_q_values)), axis=1)

    td_target = reward + gamma * target_q_value * (1. - done)
    td_error = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

soft_q_learning_loss_vmap = jax.vmap(soft_q_learning_loss, in_axes=(0, 0, 0, 0, 0), out_axes=0)

@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(params):
        predicted_q_values = dqn_module.apply({'params': params}, batch[0])
        target_q_values = dqn_module.apply({'params': target_q_network_params}, batch[3])

        # Calculate target based on log-sum-exp
        target_v = jax.scipy.special.logsumexp(target_q_values / tau, axis=1) - jnp.log(n_actions)
        target_q_value = tau * target_v
        # The above 2 lines are used instead of:
        # target_q_values = jax.nn.softmax(target_q_values / tau)
        # target_q = jnp.sum(target_q_values * (target_q_values - tau * jnp.log(target_q_values)), axis=1)

        return jnp.mean(
            soft_q_learning_loss_vmap(
                predicted_q_values,
                target_q_value,
                batch[1],
                batch[2],
                batch[4]
            )
        )

    loss, gradients = jax.value_and_grad(loss_fn)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params = optax.apply_updates(q_network_params, updates)
    current_td_errors = td_error_batch(q_network_params, target_q_network_params, batch)
    return q_network_params, opt_state, loss, current_td_errors


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        while True:
            global_steps += 1

            if global_steps < 1000:  # Encourage exploration for the first 1000 steps
                action = env.action_space.sample()
            else:
                policy_probabilities = policy(q_network_params, state)
                action = jax.random.choice(jax.random.PRNGKey(global_steps), n_actions, p=policy_probabilities)
                action = int(action) 

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            temporal_difference = float(td_error_batch(
                q_network_params,
                target_q_network_params,
                (
                    jnp.asarray([state    ]),
                    jnp.asarray([action   ]),
                    jnp.asarray([reward   ]),
                    jnp.asarray([new_state])
                )
            )[0])
            per_memory.add(temporal_difference, (state, action, reward, new_state, float(done)))

            if len(per_memory.tree.data) > batch_size:
                batch = per_memory.sample(batch_size)
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

            if global_steps % sync_steps == 0:
                target_q_network_params = q_network_params
                if debug:
                    print("Updated target network weights")

            if debug_render:
                env.render()

            if done:
                print(f"{episode} - Total reward: {sum(episode_rewards)}")
                break
finally:
    env.close()