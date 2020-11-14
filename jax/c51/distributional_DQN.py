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


debug_render  = True
num_episodes  = 500
batch_size    = 64
learning_rate = 0.01
sync_steps    = 100
memory_length = 5000

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
z_holder = [v_min + i*dz for i in range(n_atoms)]

nn_module = DistributionalDQN.partial(n_actions=n_actions, n_atoms=n_atoms)
_, params = nn_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])

nn        = flax.nn.Model(nn_module, params)
target_nn = flax.nn.Model(nn_module, params)



@jax.jit
def inference(model, input_batch):
    return model(input_batch)



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
                action = env.action_space.sample()
                pass
            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
            new_state, reward, done, _ = env.step(int(action))

            temporal_difference = 1.0
            per_memory.add(temporal_difference, (state, action, reward, new_state, int(done)))

            batch = per_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for i in range(batch_size):
                states.append     (batch[i][1][0])
                actions.append    (batch[i][1][1])
                rewards.append    (batch[i][1][2])
                next_states.append(batch[i][1][3])
                dones.append      (batch[i][1][4])

            z            = inference(nn       , jnp.array(next_states))
            z_           = inference(target_nn, jnp.array(next_states))
            z_concat     = jnp.vstack(z)
            q            = jnp.sum(jnp.multiply(z_concat, jnp.array(z_holder)), axis=1)
            q            = q.reshape((batch_size, n_actions), order='F')
            next_actions = np.argmax(q, axis=1)
            print(next_actions)

            #print(z)
            #print(z_)


            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0:
                pass

            if debug_render:
                env.render()

            if done:
                print("{} episode, reward : {}".format(episode, sum(episode_rewards)))
                break

finally:
    env.close()
