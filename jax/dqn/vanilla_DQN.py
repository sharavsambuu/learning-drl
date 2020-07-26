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
debug         = False
num_episodes  = 200
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000
replay_memory = deque(maxlen=memory_length)

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99 # discount factor


class DeepQNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_actions)
        return output_layer


env   = gym.make('CartPole-v1')
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

@jax.vmap
def q_learning_loss(q_value_vec, target_q_value_vec, action, reward, done):
    td_target = reward + gamma*jnp.amax(target_q_value_vec)*(1.-done)
    td_error  = jax.lax.stop_gradient(td_target) - q_value_vec[action]
    return jnp.square(td_error)

@jax.jit
def train_step(optimizer, target_model, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        predicted_q_values = model(batch[0])
        target_q_values    = target_model(batch[3])
        return jnp.mean(
                q_learning_loss(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss


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
                action, q_values = policy(optimizer.target, state)
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
                if debug:
                    #print("epsilon :", epsilon)
                    pass

            new_state, reward, done, _ = env.step(int(action))

            replay_memory.append((state, action, reward, new_state, int(done)))

            # Хангалттай batch цугласан бол DQN сүлжээг сургах
            if (len(replay_memory)>batch_size):
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                optimizer, loss = train_step(
                                            optimizer,
                                            target_q_network,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур 
                                                # төхөөрөмийн санах ойруу хуулах
                                                jnp.asarray(states),
                                                jnp.asarray(actions),
                                                jnp.asarray(rewards),
                                                jnp.asarray(next_states),
                                                jnp.asarray(dones)
                                            )
                                        )

            episode_rewards.append(reward)
            state = new_state

            # Тодорхой алхам тутамд target неорон сүлжээний жингүүдийг сайжирсан хувилбараар солих
            if global_steps%sync_steps==0:
                target_q_network = target_q_network.replace(params=optimizer.target.params)
                if debug:
                    print("сайжруулсан жингүүдийг target неорон сүлжээрүү хууллаа")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
