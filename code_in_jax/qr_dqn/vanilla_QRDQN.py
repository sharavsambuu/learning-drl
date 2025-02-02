#
# QR-DQN aka Quantile Regression DQN which is improvements over C51
#
#   QR-DQN is like upgrading from a histogram-based view of value distributions (C51) to a 
#   more flexible and adaptable quantile-based view. By learning to predict quantiles directly 
#   using quantile regression, QR-DQN aims to capture a richer and more accurate representation 
#   of the uncertainty and shape of the value distribution, potentially leading to 
#   improved RL performance.
#
#


import os
import random
import math
import matplotlib
matplotlib.use('TkAgg')  # try 'QtAgg' if TkAgg doesn't work
import matplotlib.pyplot as plt
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax

debug_render  = True
num_episodes  = 500
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 1000
epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01
gamma         = 0.99

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
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class QRDQN(nn.Module):
    n_actions: int
    n_quantiles: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions * self.n_quantiles)(x)
        x = jnp.reshape(x, (-1, self.n_actions, self.n_quantiles))
        return x

per_memory = PERMemory(memory_length)

env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n

n_quantiles = 50
tau         = jnp.linspace(0.0, 1.0, n_quantiles + 1)
tau_hat     = (tau[:-1] + tau[1:]) / 2

nn_module   = QRDQN(n_actions=n_actions, n_quantiles=n_quantiles)
dummy_input = jnp.zeros(state.shape)
params      = nn_module.init(jax.random.PRNGKey(0), dummy_input)['params']

nn_model_params        = params
target_nn_model_params = params

optimizer_def   = optax.adam(learning_rate)
optimizer_state = optimizer_def.init(nn_model_params)

@jax.jit
def inference(params, input_batch):
    return nn_module.apply({'params': params}, input_batch)

@jax.jit
def backpropagate(optimizer_state, model_params, target_model_params, states, actions, rewards, next_states, dones):
    def loss_fn(params):
        current_quantiles = nn_module.apply({'params': params}, states)
        target_quantiles = nn_module.apply({'params': target_model_params}, next_states)

        target_q_values = jnp.mean(target_quantiles, axis=2)
        best_next_actions = jnp.argmax(target_q_values, axis=1)

        target_quantiles_best_actions = target_quantiles[jnp.arange(batch_size), best_next_actions]

        u = rewards[:, None] + gamma * (1 - dones[:, None]) * target_quantiles_best_actions

        current_quantiles_actions = current_quantiles[jnp.arange(batch_size), actions]

        loss_component = jax.vmap(quantile_huber_loss_f, in_axes=(0, 0))(current_quantiles_actions, u)
        loss = jnp.mean(loss_component)
        return loss

    loss, gradients = jax.value_and_grad(loss_fn)(model_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, model_params)
    new_model_params = optax.apply_updates(model_params, updates)
    return new_optimizer_state, new_model_params, loss

@jax.jit
def quantile_huber_loss_f(quantiles, target_quantiles):
    u = jnp.expand_dims(target_quantiles, axis=-1) - jnp.expand_dims(quantiles, axis=0)
    huber_loss = jnp.where(jnp.abs(u) < 1.0, 0.5 * u**2, jnp.abs(u) - 0.5)
    loss = huber_loss * jnp.abs(jnp.expand_dims(tau_hat, axis=1) - jnp.where(u < 0, 1.0, 0.0))
    return jnp.sum(loss) # no need to specify axis here, since we want to sum over all quantiles

if debug_render:
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

plt.ion()

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
                outputs = inference(nn_model_params, jnp.array([state]))
                if debug_render:
                    try:
                        plt.clf()
                        for action_index in range(n_actions):
                            plt.plot(jnp.squeeze(outputs)[action_index], tau_hat, label=f'Action {action_index}')
                        plt.xlabel("Quantile Value")
                        plt.ylabel("Tau")
                        plt.title("Quantile Regression DQN Value Distribution")
                        plt.legend(loc='upper left')
                        fig.canvas.draw()
                        plt.show(block=False)
                        plt.pause(0.001)
                    except Exception as e:
                        print(f"Error during plotting: {e}")

                q_values = jnp.mean(outputs, axis=2)
                action = int(jnp.argmax(q_values[0]))

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            target_quantiles_next_state = inference(target_nn_model_params, jnp.array([new_state]))
            predicted_quantiles_current_state = inference(nn_model_params, jnp.array([state]))

            best_next_action = jnp.argmax(jnp.mean(target_quantiles_next_state, axis=2)[0])
            target_quantiles_next_state_best_action = target_quantiles_next_state[0][best_next_action]

            u = reward + gamma * (1 - done) * target_quantiles_next_state_best_action
            td_error = jnp.mean(jnp.abs(u - predicted_quantiles_current_state[0][action]))

            per_memory.add(td_error, (state, action, reward, new_state, int(done)))

            batch = per_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            idxs = []
            for i in range(batch_size):
                idxs       .append(batch[i][0])
                states     .append(batch[i][1][0])
                actions    .append(batch[i][1][1])
                rewards    .append(batch[i][1][2])
                next_states.append(batch[i][1][3])
                dones      .append(batch[i][1][4])

            optimizer_state, nn_model_params, loss = backpropagate(
                optimizer_state, nn_model_params, target_nn_model_params,
                jnp.array(states     ),
                jnp.array(actions    ),
                jnp.array(rewards    ),
                jnp.array(next_states),
                jnp.array(dones      )
            )

            new_priorities = np.zeros(batch_size)
            for i in range(batch_size):
                predicted_quantiles_batch = inference(nn_model_params, jnp.array([states[i]]))
                target_quantiles_batch = inference(target_nn_model_params, jnp.array([next_states[i]]))
                best_next_action_batch = jnp.argmax(jnp.mean(target_quantiles_batch, axis=2)[0])
                target_quantile_batch_best_action = target_quantiles_batch[0][best_next_action_batch]
                u = rewards[i] + gamma * (1 - dones[i]) * target_quantile_batch_best_action
                td_error_sample = jnp.mean(jnp.abs(u - predicted_quantiles_batch[0][actions[i]]))
                new_priorities[i] = td_error_sample

            for i in range(batch_size):
                per_memory.update(idxs[i], new_priorities[i])

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_nn_model_params = nn_model_params

            if debug_render:
                env.render()

            if done:
                print("{} episode, reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()