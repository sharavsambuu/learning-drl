import os
import random
import math
import time
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np
import pybullet_envs


debug_render  = True
num_episodes  = 500
batch_size    = 128
learning_rate = 0.001
sync_steps    = 1
memory_length = 4000

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99  # discount factor
alpha         = 0.6   # entropy tradeoff factor
tau           = 0.005 # soft update


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


class GaussianPolicy(flax.nn.Module):
    def apply(self, x, n_actions, key=None, sample=False, clip_min=-1., clip_max=1.):
        policy_layer  = CommonNetwork(x, n_actions*2)
        mean, log_std = jnp.split(policy_layer, 2, axis=-1)
        log_std       = jnp.clip(log_std, clip_min, clip_max)
        if sample:
            stds      = jnp.exp(log_std)
            xs        = gaussian_normal(key, mean, stds)
            actions   = flax.nn.tanh(xs)
            log_probs = log_prob(mean, stds, xs) - jnp.log(1-jnp.square(actions)+1e-6)
            entropies = -jnp.sum(log_probs, axis=1, keepdims=True)
            return actions, entropies, flax.nn.tanh(mean)
        else:
            return mean, log_std



environment_name = 'CartPole-v1' 
env              = gym.make(environment_name)

if debug_render:
    env.render(mode="human")
state = env.reset()


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
                action = env.action_space.sample()

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            next_state, reward, done, _ = env.step(action)


            if global_steps%sync_steps==0:
                pass


            episode_rewards.append(reward)
            state = next_state

            if debug_render:
                #time.sleep(1. / 60)
                env.render(mode="human")

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
