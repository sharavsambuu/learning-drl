import os
import sys
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
    def apply(self, x, n_actions):
        q1 = CommonNetwork(x, n_actions)
        q2 = CommonNetwork(x, n_actions)
        return q1, q2

class Policy(flax.nn.Module):
    def apply(self, x, n_actions):
        action_logits = CommonNetwork(x, n_actions)
        output_layer  = flax.nn.softmax(action_logits)
        return output_layer


environment_name = 'CartPole-v1'
env              = gym.make(environment_name)
state            = env.reset()


# неорон сүлжээнүүдийг үүсгэх инференс хийж харах тестүүд
critic_module = TwinQNetwork.partial(n_actions=env.action_space.n)
_, params     = critic_module.init_by_shape(
        jax.random.PRNGKey(0),
        [state.shape]
        )

critic        = flax.nn.Model(critic_module, params)
target_critic = flax.nn.Model(critic_module, params)

actor_module    = Policy.partial(n_actions=env.action_space.n)
_, actor_params = actor_module.init_by_shape(
        jax.random.PRNGKey(0),
        [state.shape]
        )
actor           = flax.nn.Model(actor_module, actor_params)


critic_optimizer = flax.optim.Adam(learning_rate).create(critic)
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor)

print("tests done.")
sys.exit(0)

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
