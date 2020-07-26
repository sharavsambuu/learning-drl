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
num_episodes  = 200
learning_rate = 0.001
gamma         = 0.99 # discount factor


class PolicyNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer


env   = gym.make('CartPole-v0')
state = env.reset()

n_actions        = env.action_space.n

pg_module        = PolicyNetwork.partial(n_actions=n_actions)
_, params        = pg_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
policy_network   = flax.nn.Model(pg_module, params)

optimizer        = flax.optim.Adam(learning_rate).create(policy_network)


global_steps = 0
try:
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        while True:
            global_steps = global_steps+1
            action       = env.action_space.sample()

            new_state, reward, done, _ = env.step(int(action))

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = new_state

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(rewards)))
                break
finally:
    env.close()
