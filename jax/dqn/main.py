import os
import random
import math
import gym
from collections import deque

import flax
import jax

debug_render  = True
num_episodes  = 100
memory_length = 4000
batch_size    = 64
learning_rate = 0.001
replay_memory = deque(maxlen=memory_length)


class DeepQNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_actions)
        return output_layer


env   = gym.make('CartPole-v0')
state = env.reset()

n_actions        = env.action_space.n

dqn_module       = DeepQNetwork.partial(n_actions=n_actions)
_, params        = dqn_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
q_network        = flax.nn.Model(dqn_module, params)
target_q_network = flax.nn.Model(dqn_module, params)

optimizer        = flax.optim.Adam(learning_rate).create(q_network)


global_steps = 0
try:
    for episode in range(num_episodes):
        state = env.reset()
        while True:
            global_steps = global_steps+1
            action       = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            print(new_state.shape)
            if done:
                new_state = None
            replay_memory.append((state, action, reward, new_state))

            # replay логик энд бичигдэнэ

            state = new_state
            if debug_render:
                env.render()
            if done:
                break
finally:
    env.close()
