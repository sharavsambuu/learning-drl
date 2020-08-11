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


class CommonNetwork(flax.nn.Module):
    def apply(self, x, n_outputs):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_outputs)
        return output_layer

class TwinQNetwork(flax.nn.Module):
    def apply(self, x):
        q1 = CommonNetwork(x, 1)
        q2 = CommonNetwork(x, 1)
        return q1, q2

# https://github.com/google/jax/issues/2173
@jax.jit
def gaussian_normal(key, mu, sigma):
    normals = mu + jax.random.normal(key, mu.shape)*sigma
    return normals

# https://github.com/henry-prior/jax-rl/blob/master/utils.py
@jax.jit
def gaussian_likelihood(sample, mu, log_sig):
    pre_sum = -0.5 * (((sample - mu) / (jnp.exp(log_sig) + 1e-6)) ** 2 + 2 * log_sig + jnp.log(2 * jnp.pi))
    return jnp.sum(pre_sum, axis=-1)    

# https://github.com/henry-prior/jax-rl/blob/master/models.py
class GaussianPolicy(flax.nn.Module):
    def apply(self, x, n_actions, max_action, key=None, sample=False, clip_min=-1., clip_max=1.):
        policy_layer = CommonNetwork(x, n_actions*2)
        mu, log_sig  = jnp.split(policy_layer, 2, axis=-1)
        log_sig      = flax.nn.softplus(log_sig)
        log_sig      = jnp.clip(log_sig, clip_min, clip_max)
        if sample:
            sig    = jnp.exp(log_sig)
            pi     = gaussian_normal(key, mu, sig)
            log_pi = gaussian_likelihood(pi, mu, log_sig)
            pi     = flax.nn.tanh(pi)
            log_pi = log_pi - jnp.sum(jnp.log(flax.nn.relu(1-pi**2)+1e-6), axis=-1)
            return max_action*pi, log_pi
        else:
            return max_action*flax.nn.tanh(mu), log_sig



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

actor_module    = GaussianPolicy.partial(n_actions=env.action_space.shape[0], max_action=1., key=jax.random.PRNGKey(0))
_, actor_params = actor_module.init_by_shape(
    jax.random.PRNGKey(0),
    [env.observation_space.shape])
actor           = flax.nn.Model(actor_module, actor_params)


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

print("####### critic inference test :")
q1, q2 = critic(test_input)
print(q1.shape)
print(q2.shape)
print(q1)

print("####### actor inference test :")
actions, log_sig = actor(test_state)
print("actions :")
print(actions)
print("log_sig :")
print(log_sig)

print("####### actor sampling test :")
actions, log_pi = actor(test_state, key=jax.random.PRNGKey(0), sample=True)
print("sampling actions :")
print(actions)
print("log_pi :")
print(log_pi)


#print("DONE.")
#exit(0)


@jax.jit
def actor_inference(model, state, key):
    actions, _ = model(test_state, key=key, sample=True)
    return actions


@jax.jit
def critic_inference(model, state, action):
    state_action = jnp.concatenate((state, action), axis=-1)
    q1, q2       = model(state_action)
    return q1, q2

@jax.jit
def target_critic_inference(actor_model, target_critic_model, next_state, reward, done, alpha, key):
    next_actions, next_entropies = actor_model(next_state, key=key, sample=True)
    next_state_action            = jnp.concatenate((next_state, next_actions), axis=-1)
    next_q1, next_q2             = target_critic_model(next_state_action)
    next_q                       = jnp.min([next_q1, next_q2])+alpha*next_entropies
    target_q                     = reward+(1.0-done)*gamma*next_q
    return target_q


per_memory   = PERMemory(memory_length)
rng          = jax.random.PRNGKey(0)
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
                rng, new_key = jax.random.split(rng)
                action = actor_inference(actor, state, new_key)

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            next_state, reward, done, _ = env.step(action)

            episode_rewards.append(reward)

            q1, q2 = critic_inference(
                critic, 
                jnp.asarray([state]),
                jnp.asarray([action])
                )
            rng, new_key = jax.random.split(rng)
            target_q     = target_critic_inference(
                actor, 
                target_critic, 
                jnp.asarray([next_state]), 
                jnp.asarray([reward]), 
                jnp.asarray([int(done)]), 
                1.0, 
                new_key)
            print("q1 value :")
            print(q1)
            print("target q value :")
            print(target_q)
            td_error = jnp.abs(q1[0]-target_q)[0]
            print("td error :")
            print(td_error)
            per_memory.add(td_error, (state, action, reward, next_state, int(done)))


            state = next_state

            if debug_render:
                time.sleep(1. / 60)
                env.render(mode="human")

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
