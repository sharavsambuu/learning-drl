import os
import sys
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
memory_length = 1000  # Increased memory length for better stability

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99


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


class DistributionalDQN(flax.nn.Module):
    def apply(self, x, n_actions, n_atoms):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        outputs = []
        for _ in range(n_actions):
            atom_layer      = flax.nn.Dense(activation_layer_2, n_atoms)
            atom_activation = flax.nn.softmax(atom_layer)
            outputs.append(atom_activation)
        return outputs


per_memory = PERMemory(memory_length)

env        = gym.make('CartPole-v1')
state      = env.reset()
n_actions  = env.action_space.n


v_min    = -10.0
v_max    = 10.0
n_atoms  = 51
dz       = float(v_max-v_min)/(n_atoms-1)
z_holder = jnp.array([v_min + i*dz for i in range(n_atoms)]) # Use JAX array

nn_module = DistributionalDQN.partial(n_actions=n_actions, n_atoms=n_atoms)
_, params = nn_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])

nn        = flax.nn.Model(nn_module, params)
target_nn = flax.nn.Model(nn_module, params)

optimizer = flax.optim.Adam(learning_rate).create(nn)


@jax.jit
def inference(model, input_batch):
    outputs = model(input_batch)
    # outputs is a list of n_actions arrays, each (n_atoms,)
    # Convert list of arrays to a single array (n_actions, n_atoms)
    return jnp.stack(outputs)


@jax.jit
def backpropagate(optimizer, model, target_model, states, actions, rewards, next_states, dones):
    def loss_fn(model):
        predicted_distributions = inference(model, states) # (batch_size, n_actions, n_atoms)
        target_distributions  = inference(target_model, next_states) # (batch_size, n_actions, n_atoms)

        # Calculate Quantile Regression Huber loss for each action in the batch
        loss = jnp.mean(
            jax.vmap(categorical_loss_fn)(
                predicted_distributions, # Predicted distributions for all actions in current state
                target_distributions,    # Target distributions for all actions in next state
                actions,                 # Actions taken in current states
                rewards,                 # Rewards received
                dones                    # Done flags
            )(jnp.arange(batch_size))    # Vmap over batch dimension
        )
        return loss

    gradients, loss = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

@jax.jit # Calculate categorical cross entropy loss for a single sample in the batch
def categorical_loss_fn(predicted_distributions, target_distributions, action, reward, done, index):
    # Get predicted distribution for the taken action: (n_atoms,)
    predicted_action_dist = predicted_distributions[index, action]
    # Calculate the projected distribution for the taken action: (n_atoms,)
    projected_dist = projection_distribution(index, reward, done, target_distributions, action)
    # Categorical cross-entropy loss
    return -jnp.sum(projected_dist * jnp.log(predicted_action_dist + 1e-6)) # Adding small epsilon for numerical stability

@jax.jit # Projection function (C51 projection)
def projection_distribution(index, reward, done, next_state_distributions, action):
    # Initialize projection distribution to zero: (n_atoms,)
    projected_distribution = jnp.zeros((n_atoms,))

    if done: # Terminal state
        Tz = jnp.clip(reward, v_min, v_max)
        bj = (Tz - v_min) / dz
        lower_bound = jnp.int32(jnp.floor(bj))
        upper_bound = jnp.int32(jnp.ceil(bj))
        fraction_upper = bj - lower_bound
        fraction_lower = 1.0 - fraction_upper

        projected_distribution = projected_distribution.at[lower_bound].add(fraction_lower)
        projected_distribution = projected_distribution.at[upper_bound].add(fraction_upper)
    else: # Non-terminal state
        next_state_dist = next_state_distributions[index] # (n_actions, n_atoms)
        # Compute expected Q-values for next state using target network: (n_actions,)
        q_values = jnp.sum(next_state_dist * z_holder, axis=1)
        # Select best action based on expected Q-values: scalar
        best_next_action = jnp.argmax(q_values)
        # Get the distribution of the best next action: (n_atoms,)
        next_state_best_action_dist = next_state_dist[best_next_action]

        for atom_index in range(n_atoms):
            Tz = jnp.clip(reward + gamma * z_holder[atom_index], v_min, v_max)
            bj = (Tz - v_min) / dz
            lower_bound = jnp.int32(jnp.floor(bj))
            upper_bound = jnp.int32(jnp.ceil(bj))
            fraction_upper = bj - lower_bound
            fraction_lower = 1.0 - fraction_upper

            projected_distribution = projected_distribution.at[lower_bound].add(fraction_lower * next_state_best_action_dist[atom_index])
            projected_distribution = projected_distribution.at[upper_bound].add(fraction_upper * next_state_best_action_dist[atom_index])
    return projected_distribution


# for visualizing atoms
fig = plt.gcf()
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
                outputs  = inference(nn, jnp.array([state])) # (n_actions, n_atoms)
                if debug_render:
                    plt.clf()
                    plt.bar(z_holder, outputs[0], alpha = 0.5, label='action 1', color='red') # outputs[0] is distribution for action 0
                    plt.bar(z_holder, outputs[1], alpha = 0.5, label='action 2', color='black') # outputs[1] is distribution for action 1
                    plt.legend(loc='upper left')
                    fig.canvas.draw()
                    pass
                q        = jnp.sum(outputs * z_holder, axis=1) # Expected Q-values: (n_actions,)
                action   = int(jnp.argmax(q))

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
            new_state, reward, done, _ = env.step(int(action))

            # Calculate TD Error for PER - using Categorical Cross Entropy as a proxy for error
            target_distributions_next_state = inference(target_nn, jnp.array([new_state])) # (1, n_actions, n_atoms)
            predicted_distributions_current_state = inference(nn, jnp.array([state])) # (1, n_actions, n_atoms)

            td_error = float(categorical_loss_fn(
                predicted_distributions_current_state[0], # Take distribution for the single state (n_actions, n_atoms)
                target_distributions_next_state[0], # Take distribution for the single next state (n_actions, n_atoms)
                action,
                reward,
                done,
                0 # index 0 for single sample
            ))

            per_memory.add(td_error, (state, action, reward, new_state, int(done)))

            # Batch ийн хэмжээгээр дээжүүд бэлтгэх
            batch = per_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            idxs = [] # Indices for PER update
            for i in range(batch_size):
                idxs.append      (batch[i][0]) # SumTree index
                states.append     (batch[i][1][0])
                actions.append    (batch[i][1][1])
                rewards.append    (batch[i][1][2])
                next_states.append(batch[i][1][3])
                dones.append      (batch[i][1][4])

            # Train step - Backpropagation
            optimizer, loss = backpropagate(
                optimizer, nn, target_nn,
                jnp.array(states),
                jnp.array(actions),
                jnp.array(rewards),
                jnp.array(next_states),
                jnp.array(dones)
            )


            # Update priorities in PER memory
            predicted_distributions_batch = inference(nn, jnp.array(states)) # (batch_size, n_actions, n_atoms)
            new_priorities = np.zeros(batch_size)
            for i in range(batch_size):
                target_distributions_batch = inference(target_nn, jnp.array([next_states[i]])) # (1, n_actions, n_atoms)
                td_error_sample = float(categorical_loss_fn(
                    predicted_distributions_batch[i], # Distribution for the i-th state in batch (n_actions, n_atoms)
                    target_distributions_batch[0], # Distribution for the i-th next state (n_actions, n_atoms)
                    actions[i],
                    rewards[i],
                    dones[i],
                    0 # dummy index as vmap is removed
                ))
                new_priorities[i] = td_error_sample


            for i in range(batch_size):
                per_memory.update(idxs[i], new_priorities[i])


            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0:
                target_nn = target_nn.replace(params=optimizer.target.params)
                #print("Copied online weights to the target neural network.")
                pass

            if debug_render:
                env.render()

            if done:
                print("{} episode, reward : {}".format(episode, sum(episode_rewards)))
                break

finally:
    env.close()