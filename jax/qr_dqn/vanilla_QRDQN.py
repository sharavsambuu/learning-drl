#
# QR-DQN which is improvements over C51
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
import gym
from collections import deque

import flax
import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

debug_render  = False 
num_episodes  = 500
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000
n_quantiles   = 200     # Number of quantiles for QR-DQN

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99    # discount factor


class SumTree: # PER Memory 
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

class PERMemory: # PER Memory 
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


class QRNetwork(flax.nn.Module): # QR-DQN Network - Output Quantile Values
    def apply(self, x, n_actions, n_quantiles): # Output quantiles, not probabilities
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_actions * n_quantiles) # Output n_actions * n_quantiles values
        return output_layer.reshape((n_actions, n_quantiles)) # Reshape to (n_actions, n_quantiles)


env   = gym.make('CartPole-v1')
state = env.reset()

n_actions        = env.action_space.n

qr_dqn_module    = QRNetwork.partial(n_actions=n_actions, n_quantiles=n_quantiles) # Use QRNetwork
_, params        = qr_dqn_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
q_network        = flax.nn.Model(qr_dqn_module, params)
target_q_network = flax.nn.Model(qr_dqn_module, params) # Target network is also QRNetwork

optimizer        = flax.optim.Adam(learning_rate).create(q_network)

per_memory       = PERMemory(memory_length)

tau_vals         = jnp.array([(i + 0.5) / n_quantiles for i in range(n_quantiles)]) # Quantile midpoints for loss calculation

@jax.jit
def policy(model, x): # Policy remains same - choose action with highest expected Q-value
    quantile_values = model(x) # Get quantile values
    expected_q_values = jnp.mean(quantile_values, axis=1) # Expected Q-values are mean of quantiles
    max_q_action       = jnp.argmax(expected_q_values) # Choose action with max expected Q-value
    return max_q_action, expected_q_values

@jax.jit
def calculate_td_error(q_value_vec, target_q_value_vec, action, reward): # TD error calculation - now based on quantiles
    # q_value_vec: (n_quantiles,) - Quantile values for the taken action
    # target_q_value_vec: (n_actions, n_quantiles) - Target quantile values for all actions in next state
    expected_next_q_value = jnp.mean(target_q_value_vec, axis=1) # Expected Q-values for next state
    td_target             = reward + gamma*jnp.max(expected_next_q_value) # Expected next Q-value as target
    td_error              = td_target - jnp.mean(q_value_vec) # TD error based on expected Q-value
    return jnp.abs(td_error)

@jax.jit
def td_error(model, target_model, batch): # TD error function - now using calculate_td_error for QR-DQN
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    predicted_quantile_values = model(batch[0]) # (batch_size, n_actions, n_quantiles)
    target_quantile_values    = target_model(batch[3]) # (batch_size, n_actions, n_quantiles)

    # Get quantile values for the taken actions: (batch_size, n_quantiles)
    predicted_action_quantile_values = predicted_quantile_values[jnp.arange(batch_size), batch[1]]

    return calculate_td_error(predicted_action_quantile_values, target_quantile_values, batch[1], batch[2]) # Use adjusted calculate_td_error

@jax.vmap # vmap for Pinball loss calculation
def pinball_loss(predicted_quantiles, target_quantile_value): # Pinball loss function for Quantile Regression
    # predicted_quantiles: (n_quantiles,) - Predicted quantile values for a single action
    # target_quantile_value: scalar - Target expected Q-value (using max Q-value from target network)

    # Reshape for broadcasting: (n_quantiles, 1) and (1, n_quantiles)
    predicted_quantiles_reshaped = predicted_quantiles.reshape((-1, 1))
    target_quantile_value_reshaped = jnp.reshape(target_quantile_value, (1, -1)) # Target is scalar, reshape to (1, n_quantiles) for vmap

    # Quantile midpoints tau_vals: (n_quantiles,) - Needs to be broadcastable
    tau = tau_vals.reshape((-1, 1)) # Reshape tau_vals for broadcasting

    # Indicator function for positive/negative error: (n_quantiles, n_quantiles)
    indicator = ((target_quantile_value_reshaped - predicted_quantiles_reshaped) < 0).astype(jnp.float32)

    # Pinball loss calculation: (n_quantiles, n_quantiles) - then sum over quantiles to get a scalar loss for each sample
    loss = jnp.abs(tau - indicator) * jnp.abs(target_quantile_value_reshaped - predicted_quantiles_reshaped)
    return jnp.sum(loss, axis=-1) # Sum over quantiles to get a scalar loss


@jax.jit
def train_step(optimizer, target_model, batch): # Train step function - now using Pinball loss
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        predicted_quantile_values = model(batch[0]) # (batch_size, n_actions, n_quantiles)
        target_quantile_values    = target_model(batch[3]) # (batch_size, n_actions, n_quantiles)
        expected_next_q_values    = jnp.mean(target_quantile_values, axis=2) # Expected Q-values for next state (batch_size, n_actions)
        max_next_q_values         = jnp.max(expected_next_q_values, axis=1) # Max expected Q-values for next state (batch_size,)

        # Calculate Pinball loss - vmap over batch
        loss = jnp.mean(
            pinball_loss(
                predicted_quantile_values[jnp.arange(batch_size), batch[1]], # Predicted quantiles for taken actions (batch_size, n_quantiles)
                batch[2] + gamma * (1 - batch[4]) * max_next_q_values # Target expected Q-values (batch_size,) - broadcasted to (batch_size, n_quantiles) by vmap
            )
        )
        return loss
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error(optimizer.target, target_model, batch) # Use QR-DQN TD error


fig = plt.gcf() # Visualization setup (same as C51 - optional)
fig.show()
fig.canvas.draw()


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                distribution, q_values = policy(optimizer.target, jnp.asarray([state])) # Policy is same, but now uses QR-DQN network
                if debug_render:
                    plt.clf() # Visualization (optional)
                    # Assuming n_quantiles is reasonably small for visualization, otherwise, it will be too many bars
                    for i in range(n_actions):
                        plt.bar(z_holder, distribution[i], alpha = 0.5, label=f'action {i+1}') # You might need to adjust z_holder for quantiles or create a different x-axis
                    plt.legend(loc='upper left')
                    fig.canvas.draw()
                    pass
                action = np.array(action)


            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            new_state, reward, done, _ = env.step(int(action))

            # sample нэмэхдээ temporal difference error-ийг тооцож нэмэх (using QR-DQN TD error)
            temporal_difference = float(td_error(optimizer.target, target_q_network, (
                    jnp.asarray([state]),
                    jnp.asarray([action]),
                    jnp.asarray([reward]),
                    jnp.asarray([new_state])
                ))[0])
            per_memory.add(temporal_difference, (state, action, reward, new_state, int(done)))

            if len(per_memory) > batch_size:
                # Prioritized Experience Replay санах ойгоос batch үүсгээд DQN сүлжээг сургах (using QR-DQN)
                batch = per_memory.sample(batch_size)
                states, actions, rewards, next_states, dones = [], [], [], [], []
                for i in range(batch_size):
                    states.append     (batch[i][1][0])
                    actions.append    (batch[i][1][1])
                    rewards.append    (batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones.append      (batch[i][1][4])

                optimizer, loss, new_td_errors = train_step( # Train step using QR-DQN train_step
                                            optimizer,
                                            target_q_network,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур
                                                # төхөөрөмжийн санах ойруу хуулах (QR-DQN batch)
                                                jnp.asarray(states),
                                                jnp.asarray(actions),
                                                jnp.asarray(rewards),
                                                jnp.asarray(next_states),
                                                jnp.asarray(dones)
                                            )
                                        )
                # batch-аас бий болсон temporal difference error-ийн дагуу санах ойг шинэчлэх (using QR-DQN TD error)
                new_td_errors = np.array(new_td_errors)
                for i in range(batch_size):
                    idx = batch[i][0]
                    per_memory.update(idx, new_td_errors[i])

            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0:
                target_q_network = target_q_network.replace(params=optimizer.target.params)
                pass

            if debug_render:
                env.render()

            if done:
                print("{} episode, reward : {}".format(episode, sum(episode_rewards)))
                break

finally:
    env.close()