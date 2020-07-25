import os
import random
import math
import gym
from collections import deque

import flax
import jax
from jax import numpy as jnp
import numpy as np


debug_render  = True
debug         = True
num_episodes  = 100
batch_size    = 64
learning_rate = 0.001
memory_length = 4000
replay_memory = deque(maxlen=memory_length)

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01


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


@jax.jit
def policy(model, x):
    predicted_q_values = model(x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


global_steps = 0
try:
    for episode in range(num_episodes):
        state = env.reset()
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(optimizer.target, state)
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
                if debug:
                    print("epsilon :", epsilon)

            new_state, reward, done, _ = env.step(int(action))
            #print(new_state.shape)
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
