# Soft Actor-Critic, discrete action space
# References:
#  - SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS
#    https://arxiv.org/pdf/1910.07207.pdf
#  - https://github.com/henry-prior/jax-rl
#  - https://github.com/ku2482/sac-discrete.pytorch

import os
import random
import math
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np


debug_render  = True
debug         = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99
batch_size    = 64
sync_steps    = 100
learn_steps   = 10
memory_length = 4000

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01


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


class Actor(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        dense_layer_3      = flax.nn.Dense(activation_layer_2, n_actions)
        action_logits      = flax.nn.softmax(dense_layer_3)
        return action_logits

class DoubleCritic(flax.nn.Module):
    def apply(self, state, action):
        state_action = jnp.concatenate([state, action])

        q1 = flax.nn.Dense(state_action, 64)
        q1 = flax.nn.relu(q1)
        q1 = flax.nn.Dense(q1, 32)
        q1 = flax.nn.relu(q1)
        q1 = flax.nn.Dense(q1, 1)

        q2 = flax.nn.Dense(state_action, 64)
        q2 = flax.nn.relu(q2)
        q2 = flax.nn.Dense(q2, 32)
        q2 = flax.nn.relu(q2)
        q2 = flax.nn.Dense(q2, 1)

        return q1, q2


@jax.jit
def inference_actor(model, x):
    action_logits = model(jnp.asarray([x]))[0]
    return action_logits

@jax.jit
def act_greedy(model, x):
    action_logits = model(jnp.asarray([x]))[0]
    return jnp.argmax(action_logits)


env          = gym.make('CartPole-v1')
state        = env.reset()

n_actions    = env.action_space.n
state_shape  = state.shape
action_shape = (n_actions,)

per_memory   = PERMemory(memory_length)
global_step  = 0


rng      = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)

actor_module        = Actor.partial(n_actions=n_actions)
_, actor_params     = actor_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])
actor_model         = flax.nn.Model(actor_module, actor_params)

critic_module       = DoubleCritic.partial()
_, critic_params    = critic_module.init_by_shape(jax.random.PRNGKey(0), [state_shape, action_shape])
critic_model        = flax.nn.Model(critic_module, critic_params)
target_critic_model = flax.nn.Model(critic_module, critic_params)

actor_optimizer     = flax.optim.Adam(learning_rate).create(actor_model)
critic_optimizer    = flax.optim.Adam(learning_rate).create(critic_model)


try:
    for episode in range(num_episodes):
        state           = env.reset()
        episode_rewards = []
        while True:
            global_step = global_step+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action_logits  = inference_actor(actor_optimizer.target, state)
                probabilities  = np.array(action_logits)
                probabilities /= np.sum(probabilities)
                action         = np.random.choice(n_actions, p=probabilities)

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_step)

            next_state, reward, done, _ = env.step(int(action))

            sampling_weight = 1
            per_memory.add(sampling_weight, (state, action, reward, next_state, int(done)))

            episode_rewards.append(reward)

            state = next_state

            if global_step%sync_steps==0:
                pass

            if debug_render:
                env.render()

            if done:
                print(episode, " - reward :", sum(episode_rewards))
                break
finally:
    env.close()
