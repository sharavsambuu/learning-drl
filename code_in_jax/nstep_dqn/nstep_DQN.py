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
import time

debug_render  = True
debug         = False
num_episodes  = 500
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000
n_steps       = 3      # N-step lookahead

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99   # discount factor


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
        self.tree     = SumTree(capacity)
        self.capacity = capacity
        self.size     = 0  # track how many real samples we stored
    def _get_priority(self, error):
        return (error+self.e)**self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.size = min(self.size + 1, self.capacity) 
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


env              = gym.make('CartPole-v1', render_mode="human" if debug_render else None)
state, info      = env.reset()

n_actions        = env.action_space.n

dqn_module       = DeepQNetwork(n_actions=n_actions)
dummy_input      = jnp.zeros(state.shape)
params           = dqn_module.init(jax.random.PRNGKey(0), dummy_input)

q_network_params        = params['params']
target_q_network_params = params['params']

optimizer  = optax.adam(learning_rate)
opt_state  = optimizer.init(q_network_params)

per_memory     = PERMemory(memory_length)

# store full transition pieces for n-step building
n_step_buffer  = deque(maxlen=n_steps) # (state, action, reward, next_state, done)


@jax.jit
def policy(params, x):
    predicted_q_values = dqn_module.apply({'params': params}, x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


def build_n_step_sample(n_step_buffer, gamma, n_steps):
    # Build one sample from the LEFT of the buffer:
    # returns (s0, a0, R, s_k, done_k, gamma_pow)
    n_step_state  = n_step_buffer[0][0]
    n_step_action = n_step_buffer[0][1]

    n_step_return     = 0.0
    n_step_next_state = n_step_buffer[-1][3]
    n_step_done       = 0.0
    k                 = 0

    for i, (s, a, r, ns, d) in enumerate(n_step_buffer):
        n_step_return    += (gamma**i) * float(r)
        n_step_next_state = ns
        n_step_done       = float(d)
        k = i + 1
        if d or k >= n_steps:
            break

    gamma_pow = float(gamma**k)
    return n_step_state, n_step_action, float(n_step_return), n_step_next_state, float(n_step_done), float(gamma_pow)


def calculate_n_step_td_error(q_value_vec, target_q_value_vec, action, n_step_return, done, gamma_pow):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value         = jnp.sum(one_hot_actions*q_value_vec)

    bootstrap   = gamma_pow * jnp.max(target_q_value_vec) * (1.-done)
    td_target   = n_step_return + bootstrap
    td_error    = td_target - q_value
    return jnp.abs(td_error)

calculate_n_step_td_error_vmap = jax.vmap(
    calculate_n_step_td_error,
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=0
)

def n_step_q_learning_loss(q_value_vec, target_q_value_vec, action, n_step_return, done, gamma_pow):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value         = jnp.sum(one_hot_actions*q_value_vec)

    bootstrap   = gamma_pow * jnp.max(target_q_value_vec) * (1.-done)
    td_target   = n_step_return + bootstrap
    td_error    = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

n_step_q_learning_loss_vmap = jax.vmap(
    n_step_q_learning_loss,
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=0
)

@jax.jit
def td_error_batch(q_network_params, target_q_network_params, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - n_step_returns
    # batch[3] - next_states
    # batch[4] - dones
    # batch[5] - gamma_pows (gamma**k)
    predicted_q_values = dqn_module.apply({'params': q_network_params}, batch[0])
    target_q_values    = dqn_module.apply({'params': target_q_network_params}, batch[3])
    return calculate_n_step_td_error_vmap(
        predicted_q_values,
        target_q_values,
        batch[1],
        batch[2],
        batch[4],
        batch[5]
    )

@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - n_step_returns
    # batch[3] - next_states
    # batch[4] - dones
    # batch[5] - gamma_pows
    def loss_fn(params):
        predicted_q_values = dqn_module.apply({'params': params}, batch[0])
        target_q_values    = dqn_module.apply({'params': target_q_network_params}, batch[3])
        return jnp.mean(
                n_step_q_learning_loss_vmap(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4],
                    batch[5]
                    )
                )
    loss, gradients    = jax.value_and_grad(loss_fn)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params   = optax.apply_updates(q_network_params, updates)
    current_td_errors  = td_error_batch(q_network_params, target_q_network_params, batch)
    return q_network_params, opt_state, loss, current_td_errors


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state       = np.array(state, dtype=np.float32)
        n_step_buffer.clear()

        while True:
            global_steps += 1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(q_network_params, jnp.asarray(state))
                action = int(action)

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done      = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            # push full transition
            n_step_buffer.append((state, action, reward, new_state, float(done)))

            # emit a sample only when buffer is full (fixed n-step)
            if len(n_step_buffer) == n_steps:
                n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done, gamma_pow = build_n_step_sample(
                    n_step_buffer, gamma, n_steps
                )

                temporal_difference = float(td_error_batch(
                    q_network_params,
                    target_q_network_params,
                    (
                        jnp.asarray([n_step_state     ]                   ),
                        jnp.asarray([n_step_action    ], dtype=jnp.int32  ),
                        jnp.asarray([n_step_return    ], dtype=jnp.float32),
                        jnp.asarray([n_step_next_state]                   ),
                        jnp.asarray([n_step_done      ], dtype=jnp.float32),
                        jnp.asarray([gamma_pow        ], dtype=jnp.float32),
                    )
                )[0])

                per_memory.add(temporal_difference, (n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done, gamma_pow))
                n_step_buffer.popleft()

            # if episode ends, flush remaining partial samples (k < n)
            if done:
                while len(n_step_buffer) > 0:
                    n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done, gamma_pow = build_n_step_sample(
                        n_step_buffer, gamma, n_steps
                    )

                    temporal_difference = float(td_error_batch(
                        q_network_params,
                        target_q_network_params,
                        (
                            jnp.asarray([n_step_state     ]                   ),
                            jnp.asarray([n_step_action    ], dtype=jnp.int32  ),
                            jnp.asarray([n_step_return    ], dtype=jnp.float32),
                            jnp.asarray([n_step_next_state]                   ),
                            jnp.asarray([n_step_done      ], dtype=jnp.float32),
                            jnp.asarray([gamma_pow        ], dtype=jnp.float32),
                        )
                    )[0])

                    per_memory.add(temporal_difference, (n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done, gamma_pow))
                    n_step_buffer.popleft()

            # train only after real samples exist
            if (per_memory.size >= batch_size):
                batch = per_memory.sample(batch_size)
                idxs, segment_data = zip(*batch)

                states, actions, n_step_returns, next_states, dones, gamma_pows = [], [], [], [], [], []
                for data in segment_data:
                    states        .append(data[0])
                    actions       .append(data[1])
                    n_step_returns.append(data[2])
                    next_states   .append(data[3])
                    dones         .append(data[4])
                    gamma_pows    .append(data[5])

                q_network_params, opt_state, loss, new_td_errors = train_step(
                    q_network_params,
                    target_q_network_params,
                    opt_state,
                    (
                        jnp.asarray(list(states        )                   ),
                        jnp.asarray(list(actions       ), dtype=jnp.int32  ),
                        jnp.asarray(list(n_step_returns), dtype=jnp.float32),
                        jnp.asarray(list(next_states   )                   ),
                        jnp.asarray(list(dones         ), dtype=jnp.float32),
                        jnp.asarray(list(gamma_pows    ), dtype=jnp.float32),
                    )
                )

                new_td_errors_np = np.array(new_td_errors)
                for i in range(batch_size):
                    idx = idxs[i]
                    per_memory.update(idx, float(new_td_errors_np[i]))

            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0:
                target_q_network_params = q_network_params
                if debug:
                    print("copied updated weights to the target network")

            if debug_render:
                env.render()

            if done:
                print("{} - total reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
