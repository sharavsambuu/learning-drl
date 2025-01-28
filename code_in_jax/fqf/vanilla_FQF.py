#
# FQF ak Fully Parameterized Quantile Function.
# It is improvements overs C51, QR-DQN and IQN
#
#   FQF is like taking the idea of distributional RL to its most advanced form. 
#   Instead of just estimating average values or a fixed set of quantiles, 
#   FQF learns to directly model the entire quantile function – the full shape 
#   of the value distribution. It does this by using basis functions and learning 
#   to adaptively choose the most informative quantile fractions, leading to a 
#   highly flexible, data-efficient, and powerful distributional RL algorithm. 
#   It's like learning to "draw the entire CDF curve" rather than just plotting 
#   a few points on it.
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
learning_rate = 0.0001 # Lower learning rate for FQF
sync_steps    = 100
memory_length = 4000
n_quantiles   = 32     # Number of quantiles to sample for FQF (per update step) - Can be smaller
n_tau_samples = 64     # Number of tau samples for loss calculation - More samples for better loss estimate
epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01
gamma         = 0.99   # discount factor


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



class FQFNetwork(flax.nn.Module): # FQF Network - Fully Parameterized Quantile Function
    def apply(self, x, n_actions, n_quantiles): # n_quantiles here is for output, not input taus
        # State embedding network (same as DQN/QR-DQN/IQN)
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        state_embedding    = flax.nn.relu(dense_layer_2) # State embedding: (batch_size, embedding_dim)

        # Fraction proposal network - Output quantile fractions (tau_hats) and logits
        fraction_layer   = flax.nn.Dense(state_embedding, n_actions * n_quantiles) # Output n_actions * n_quantiles logits
        fraction_logits  = fraction_layer.reshape((n_actions, n_quantiles)) # Reshape to (n_actions, n_quantiles)
        tau_hats         = flax.nn.softmax(fraction_logits) # Softmax for quantile fractions: (n_actions, n_quantiles)

        # Value computation network - Compute quantile values based on state embedding and proposed fractions
        value_layer_input= state_embedding # Input is state embedding
        value_layer      = flax.nn.Dense(value_layer_input, n_actions * n_quantiles) # Output n_actions * n_quantiles values
        quantile_values  = value_layer.reshape((n_actions, n_quantiles)) # Reshape to (n_actions, n_quantiles)

        return quantile_values, tau_hats # Output quantile values and proposed fractions


env   = gym.make('CartPole-v1')
state = env.reset()
n_actions        = env.action_space.n

fqf_module       = FQFNetwork.partial(n_actions=n_actions, n_quantiles=n_quantiles) # Use FQFNetwork
_, params        = fqf_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
q_network        = flax.nn.Model(fqf_module, params)
target_q_network = flax.nn.Model(fqf_module, params) # Target network is also FQFNetwork

optimizer        = flax.optim.Adam(learning_rate).create(q_network)

per_memory       = PERMemory(memory_length)

n_tau_prime      = n_quantiles # Number of tau_prime samples for target value estimation
tau_vals         = jnp.array([(i + 0.5) / n_quantiles for i in range(n_quantiles)]) # Fixed tau values for loss calculation (same as QR-DQN)


@jax.jit
def policy(model, x, key, n_sample_policy=32): # Policy - Still epsilon-greedy, use expected Q-value from sampled quantiles
    quantile_values, _ = model(x, n_quantiles=n_sample_policy) # Sample quantiles for policy evaluation
    expected_q_values = jnp.mean(quantile_values, axis=1) # Expected Q-values: Mean over quantiles (n_actions,)
    max_q_action    = jnp.argmax(expected_q_values) # Choose action with max expected Q-value
    return max_q_action, expected_q_values


@jax.jit
def calculate_quantile_values(model, x, key, n_quantiles_sample): # Function to sample quantiles for loss calculation
    quantile_values, tau_hats = model(x, n_quantiles=n_quantiles_sample) # Fraction logits are also output now
    return quantile_values, tau_hats # Return quantile values and fraction logits

@jax.jit
def td_error(model, target_model, batch, key): # TD error calculation - now uses FQF value estimation
    predicted_quantile_values, _ = calculate_quantile_values(model, batch[0], key, n_quantiles) # Sample quantiles for current state
    target_quantile_values, _    = calculate_quantile_values(target_model, batch[3], key, n_tau_prime) # Sample quantiles for next state (separate taus for target)

    expected_next_q_values           = jnp.mean(target_quantile_values, axis=2) # Expected next Q-values: (batch_size, n_actions) - Mean over tau_prime samples
    # Get quantile values for the taken actions: (batch_size, n_quantiles)
    predicted_action_quantile_values = predicted_quantile_values[jnp.arange(batch_size), batch[1]] # (batch_size, n_quantiles)

    # TD error based on expected next Q-values (similar to QR-DQN TD error, but using FQF value estimation)
    td_target = batch[2] + gamma*jnp.max(expected_next_q_values, axis=1)*(1.-batch[4]) # (batch_size,)
    td_error  = td_target - jnp.mean(predicted_action_quantile_values, axis=1) # (batch_size,)
    return jnp.abs(td_error) # (batch_size,)


@jax.vmap # vmap for QF Regression loss calculation
def qf_regression_loss(predicted_quantiles, target_quantile_values): # QF Regression loss function - for FQF
    # predicted_quantiles: (n_quantiles,) - Predicted quantile values for a single action
    # target_quantile_values: (n_tau_prime,) - Target quantile values (sampled from target network)

    # Reshape for pairwise difference calculation: (n_quantiles, 1) and (1, n_tau_prime)
    predicted_quantiles_reshaped = predicted_quantiles.reshape((-1, 1)) # (n_quantiles, 1)
    target_quantile_values_reshaped = target_quantile_values.reshape((1, -1)) # (1, n_tau_prime)

    # Pairwise difference between predicted and target quantiles: (n_quantiles, n_tau_prime)
    pairwise_diff = target_quantile_values_reshaped - predicted_quantiles_reshaped # (n_quantiles, n_tau_prime)

    # Indicator function: 1 if diff < 0, 0 otherwise - (n_quantiles, n_tau_prime)
    indicator     = (pairwise_diff < 0).astype(jnp.float32)

    # Quantile Huber loss (element-wise): (n_quantiles, n_tau_prime)
    huber_l       = element_wise_huber_loss(pairwise_diff) # (n_quantiles, n_tau_prime)

    # Sum over target quantiles (tau_prime) dimension, reduce_mean over current quantiles (tau)
    loss          = jnp.mean(jnp.sum(jnp.abs(tau_vals.reshape((-1, 1)) - indicator) * huber_l, axis=1), axis=0) # (,) - scalar loss
    return loss


@jax.jit
def train_step(optimizer, target_model, batch, key): # Train step function - now using QF Regression loss and sampled taus
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        predicted_quantile_values, _ = calculate_quantile_values(model, batch[0], key, n_quantiles) # Sample quantiles for current state
        target_quantile_values, _    = calculate_quantile_values(target_model, batch[3], key, n_tau_prime) # Sample quantiles for next state (different taus)

        # Get target quantile values for loss calculation - Bellman target for distributional RL
        target_values = batch[2] + gamma * (1 - batch[4]) * jnp.mean(jnp.max(target_quantile_values, axis=1), axis=1, keepdims=True) # (batch_size, 1) - Expected max Q-value as target

        # Calculate QF Regression loss - vmap over batch
        loss = jnp.mean(
            qf_regression_loss(
                predicted_quantile_values[jnp.arange(batch_size), batch[1]], # Predicted quantiles for taken actions (batch_size, n_quantiles)
                target_values # Target expected Q-values (batch_size, 1) - broadcasted by vmap
            )
        )
        return loss
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error(optimizer.target, target_model, batch, key) # Use FQF TD Error


fig = plt.gcf() # Visualization setup (same as before - optional)
fig.show()
fig.canvas.draw()


rng = jax.random.PRNGKey(0) # Random key for action sampling and training
global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        while True:
            global_steps = global_steps+1
            rng, policy_key, train_key = jax.random.split(rng, 3) # Split keys

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(optimizer.target, jnp.asarray([state]), policy_key) # Use policy function with key
                if debug_render:
                    plt.clf() # Visualization - You'll need to customize this for FQF's output format
                    pass
                action = np.array(action)


            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            new_state, reward, done, _ = env.step(int(action))

            # sample нэмэхдээ temporal difference error-ийг тооцож нэмэх (using FQF TD error)
            temporal_difference = float(td_error(optimizer.target, target_q_network, (
                    jnp.asarray([state]),
                    jnp.asarray([action]),
                    jnp.asarray([reward]),
                    jnp.asarray([new_state])
                ), train_key)[0]) # Use key for td_error
            per_memory.add(temporal_difference, (state, action, reward, new_state, int(done)))

            if len(per_memory) > batch_size:
                # Prioritized Experience Replay санах ойгоос batch үүсгээд DQN сүлжээг сургах (using FQF)
                batch = per_memory.sample(batch_size)
                states, actions, rewards, next_states, dones = [], [], [], [], []
                for i in range(batch_size):
                    states.append     (batch[i][1][0])
                    actions.append    (batch[i][1][1])
                    rewards.append    (batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones.append      (batch[i][1][4])

                rng, train_key = jax.random.split(rng) # Split key for train step
                optimizer, loss, new_td_errors = train_step( # Train step using FQF train_step, pass key
                                            optimizer,
                                            target_q_network,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур
                                                # төхөөрөмжийн санах ойруу хуулах (FQF batch)
                                                jnp.asarray(states),
                                                jnp.asarray(actions),
                                                jnp.asarray(rewards),
                                                jnp.asarray(next_states),
                                                jnp.asarray(dones)
                                            ),
                                            train_key # Pass key to train step
                                        )
                # batch-аас бий болсон temporal difference error-ийн дагуу санах ойг шинэчлэх (using FQF TD error)
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