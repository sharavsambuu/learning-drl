import os
import random
import math
import time
import gym
from collections import deque

import flax
import jax
from jax import numpy as jnp
import numpy as np

import pybullet_envs


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


class DeepQNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_actions)
        return output_layer

class TwinQNetwork(flax.nn.Module):
    def apply(self, x):
        q1 = DeepQNetwork(x, 1)
        q2 = DeepQNetwork(x, 1)
        return q1, q2


env   = gym.make('HumanoidFlagrunHarderBulletEnv-v0')
#env.render(mode="human")
state = env.reset()

# (44,)
print("observation space :")
print(env.observation_space.shape)
# (17,)
print("Action space :")
print(env.action_space.shape)
# 1
print("Action space high :")
print(env.action_space.high)
# -1
print("Action space low :")
print(env.action_space.low)

state_action_shape = (env.observation_space.shape[0]+env.action_space.shape[0],)
print("StateAction shape")
print(state_action_shape)



critic_module = TwinQNetwork.partial()
_, params     = critic_module.init_by_shape(
    jax.random.PRNGKey(0), 
    [(env.observation_space.shape[0]+env.action_space.shape[0],)])


critic        = flax.nn.Model(critic_module, params)
target_critic = flax.nn.Model(critic_module, params)


# неорон сүлжээ үүсч байгаа эсэхийг шалгах туршилтууд
test_state  = env.reset()
test_action = env.action_space.sample()
print("test state shape :")
print(test_state.shape)
print("test action shape :")
print(test_action.shape)

test_input  = jnp.concatenate((test_state, test_action))
print("test input shape :")
print(test_input.shape)

print("critic inference test :")
q1, q2 = critic(test_input)
print(q1.shape)
print(q2.shape)
print(q1)


print("DONE.")
exit(0)




per_memory = PERMemory(memory_length)

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

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            #print("action", action)
            new_state, reward, done, _ = env.step(action)

            episode_rewards.append(reward)
            state = new_state

            if debug_render:
                time.sleep(1. / 60)
                env.render(mode="human")

            if done:
                #print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
