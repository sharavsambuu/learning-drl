#
# IQN which is improvements over previous C51 and QR-DQN
#
#   IQN is like making the quantile prediction process "smarter and more efficient." 
#   Instead of always predicting the same fixed set of quantiles (QR-DQN), 
#   IQN learns a function that can generate quantile values on-demand for any quantile 
#   fraction you ask for. This "implicit quantile function" allows for a more flexible, 
#   efficient, and potentially more accurate representation of value distributions in RL. 
#   It's like having a quantile "generator" rather than just a quantile "predictor."
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
n_quantiles   = 32     # Number of quantiles to sample for IQN (per update step) - Reduced for efficiency
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


class IQNNetwork(flax.nn.Module): # IQN Network - Implicit Quantile Network
    def apply(self, x, taus, n_actions): # Takes state and quantile fractions (taus) as input
        # State embedding network (same as DQN/QR-DQN)
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        state_embedding    = flax.nn.relu(dense_layer_2) # State embedding: (batch_size, embedding_dim)

        # Quantile embedding - Cosine embedding for taus
        cos_embedding      = jnp.cos(jnp.expand_dims(taus, -1) * jnp.arange(1, 65)) # Cosine embedding: (n_quantiles, embedding_dim)
        quantile_embedding = flax.nn.Dense(cos_embedding, state_embedding.shape[-1]) # Project to state embedding dimension

        # Combine state and quantile embeddings - Element-wise product
        combined_embedding = state_embedding * quantile_embedding # (n_quantiles, embedding_dim) - Broadcasted

        # Value stream - Process combined embedding to output quantile values
        value_layer        = flax.nn.Dense(combined_embedding, 32)
        value_activation   = flax.nn.relu(value_layer)
        output_layer       = flax.nn.Dense(value_activation, n_actions) # Output quantile values: (n_quantiles, n_actions)
        return output_layer # Quantile values for each action, for each sampled tau


env   = gym.make('CartPole-v1')
state = env.reset()
n_actions        = env.action_space.n

iqn_module       = IQNNetwork.partial(n_actions=n_actions) # Use IQNNetwork
_, params        = iqn_module.init_by_shape(jax.random.PRNGKey(0), [state.shape], taus=jnp.zeros((n_quantiles,))) # Dummy taus for init
q_network        = flax.nn.Model(iqn_module, params)
target_q_network = flax.nn.Model(iqn_module, params) # Target network is also IQNNetwork

optimizer        = flax.optim.Adam(learning_rate).create(q_network)

per_memory       = PERMemory(memory_length)

@jax.jit
def policy(model, x, key, n_sample_policy=32): # Policy - Still epsilon-greedy, but Q-value estimation uses sampled quantiles
    taus          = jax.random.uniform(key, shape=(n_sample_policy,)) # Sample multiple taus for policy evaluation
    quantile_values = model(x, taus=taus) # Get quantile values for sampled taus: (n_sample_policy, n_actions)
    expected_q_values = jnp.mean(quantile_values, axis=0) # Expected Q-values: Mean over quantiles (n_actions,)
    max_q_action    = jnp.argmax(expected_q_values) # Choose action with max expected Q-value
    return max_q_action, expected_q_values

@jax.jit
def calculate_quantile_values(model, x, key, n_quantiles): # Function to sample quantiles for loss calculation
    taus             = jax.random.uniform(key, shape=(n_quantiles,)) # Sample taus for loss calculation
    quantile_values    = model(x, taus=taus) # Quantile values for sampled taus: (n_quantiles, n_actions)
    return quantile_values, taus # Return quantile values and taus


@jax.jit
def td_error(model, target_model, batch, key): # TD error calculation - now uses sampled quantiles for IQN
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    predicted_quantile_values, _ = calculate_quantile_values(model, batch[0], key, n_quantiles) # Sample quantiles for current state
    target_quantile_values, _    = calculate_quantile_values(target_model, batch[3], key, n_quantiles) # Sample quantiles for next state

    # Get quantile values for the taken actions: (batch_size, n_quantiles)
    predicted_action_quantile_values = predicted_quantile_values[jnp.arange(batch_size), batch[1]] # (batch_size, n_quantiles)
    expected_next_q_values           = jnp.mean(target_quantile_values, axis=2) # Expected next Q-values: (batch_size, n_actions)

    # Calculate TD error based on expected next Q-values (same as QR-DQN TD error)
    td_target = batch[2] + gamma*jnp.max(expected_next_q_values, axis=1)*(1.-batch[4]) # (batch_size,)
    td_error  = td_target - jnp.mean(predicted_action_quantile_values, axis=1) # (batch_size,)
    return jnp.abs(td_error) # (batch_size,)


@jax.vmap # vmap for element-wise Huber loss (more stable than Pinball for IQN in some cases)
def element_wise_huber_loss(td_errors): # Element-wise Huber loss
    return jnp.where(jnp.abs(td_errors) <= 1, 0.5 * jnp.square(td_errors), jnp.abs(td_errors) - 0.5) # Huber loss formula


@jax.jit
def train_step(optimizer, target_model, batch, key): # Train step function - now using sampled quantiles and Huber loss
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        predicted_quantile_values, taus = calculate_quantile_values(model, batch[0], key, n_quantiles) # Sample quantiles for current state
        target_quantile_values, _    = calculate_quantile_values(target_model, batch[3], key, n_quantiles) # Sample quantiles for next state
        expected_next_q_values    = jnp.mean(target_quantile_values, axis=2) # Expected next Q-values: (batch_size, n_actions)
        max_next_q_values         = jnp.max(expected_next_q_values, axis=1, keepdims=True) # Max expected next Q-values: (batch_size, 1)

        # Get predicted quantiles for taken actions: (batch_size, n_quantiles)
        predicted_action_quantile_values = predicted_quantile_values[jnp.arange(batch_size), batch[1]] # (batch_size, n_quantiles)

        # Calculate pairwise difference - target_quantile_value broadcasted for all predicted quantiles
        pairwise_diff = jnp.expand_dims(jax.lax.stop_gradient(batch[2] + gamma * (1 - batch[4]) * max_next_q_values), axis=1) - jnp.expand_dims(predicted_action_quantile_values, axis=0) # (n_quantiles, n_quantiles) - Broadcasted target

        # Indicator function and Huber loss (element-wise)
        indicator     = (pairwise_diff < 0).astype(jnp.float32) # (n_quantiles, n_quantiles)
        huber_l       = element_wise_huber_loss(pairwise_diff) # (n_quantiles, n_quantiles)
        loss          = jnp.sum(jnp.mean(jnp.abs(taus.reshape((-1, 1)) - indicator) * huber_l, axis=-1), axis=-1) # Weighted by tau and mean over quantiles

        return jnp.mean(loss) # Mean over batch

    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error(optimizer.target, target_model, batch, key) # Use IQN TD error


fig = plt.gcf() # Visualization setup (same as C51/QR-DQN - optional)
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
            rng, policy_key, train_key = jax.random.split(rng, 3) # Split keys for policy and train steps

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(optimizer.target, jnp.asarray([state]), policy_key) # Use policy function with key
                if debug_render:
                    plt.clf() # Visualization (optional)
                    # Visualization will be different for IQN, you might want to visualize sampled quantiles differently
                    # e.g., plot CDF or histogram of sampled quantiles
                    pass
                action = np.array(action)


            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            new_state, reward, done, _ = env.step(int(action))

            # sample нэмэхдээ temporal difference error-ийг тооцож нэмэх (using IQN TD error)
            temporal_difference = float(td_error(optimizer.target, target_q_network, (
                    jnp.asarray([state]),
                    jnp.asarray([action]),
                    jnp.asarray([reward]),
                    jnp.asarray([new_state])
                ), train_key)[0]) # Use key for td_error calculation
            per_memory.add(temporal_difference, (state, action, reward, new_state, int(done)))

            if len(per_memory) > batch_size:
                # Prioritized Experience Replay санах ойгоос batch үүсгээд DQN сүлжээг сургах (using IQN)
                batch = per_memory.sample(batch_size)
                states, actions, rewards, next_states, dones = [], [], [], [], []
                for i in range(batch_size):
                    states.append     (batch[i][1][0])
                    actions.append    (batch[i][1][1])
                    rewards.append    (batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones.append      (batch[i][1][4])

                rng, train_key = jax.random.split(rng) # Split key for train step
                optimizer, loss, new_td_errors = train_step( # Train step using IQN train_step, pass key
                                            optimizer,
                                            target_q_network,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур
                                                # төхөөрөмжийн санах ойруу хуулах (IQN batch)
                                                jnp.asarray(states),
                                                jnp.asarray(actions),
                                                jnp.asarray(rewards),
                                                jnp.asarray(next_states),
                                                jnp.asarray(dones)
                                            ),
                                            train_key # Pass key to train step
                                        )
                # batch-аас бий болсон temporal difference error-ийн дагуу санах ойг шинэчлэх (using IQN TD error)
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
