import os
import random
import math
import gym
from collections import deque
import flax
import jax
from jax import numpy as jnp
import numpy as np
import numpy


debug_render  = True
debug         = False
num_episodes  = 3000
learning_rate = 0.01
gamma         = 0.99 # discount factor
memory_length = 4000
replay_memory = deque(maxlen=memory_length)
batch_size    = 64


# References:
#    - https://github.com/google/flax/blob/master/examples/vae/train.py
#    - https://colab.research.google.com/drive/1UCdLb5TNlOu_H6v9Pdeh5V6bIW7SPEDu

class ActorNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer        = flax.nn.Dense(x, 32)
        activation_layer   = flax.nn.relu(dense_layer)
        output_dense_layer = flax.nn.Dense(activation_layer, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(flax.nn.Module):
    def apply(self, x):
        dense_layer        = flax.nn.Dense(x, 32)
        activation_layer   = flax.nn.relu(dense_layer)
        output_layer       = flax.nn.Dense(activation_layer, 1)
        return output_layer

class A2C(flax.nn.Module):
    def apply(self, x, n_actions):
        common_dense_layer      = flax.nn.Dense(x, 64)
        common_activation_layer = flax.nn.relu(common_dense_layer)
        actor_layer             = ActorNetwork(
                common_activation_layer,
                n_actions=n_actions,
                name='actor_layer'
                )
        critic_layer            = CriticNetwork(
                common_activation_layer,
                name='critic_layer'
                )
        return actor_layer, critic_layer



env   = gym.make('CartPole-v1')
state = env.reset()

n_actions        = env.action_space.n

a2c_module       = A2C.partial(n_actions=n_actions)
_, a2c_params    = a2c_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
a2c_network      = flax.nn.Model(a2c_module, a2c_params)
a2c_optimizer    = flax.optim.Adam(learning_rate).create(a2c_network)


@jax.jit
def a2c_inference(model, x):
    action_probabilities_list, critic_value_list = model(x)
    return action_probabilities_list, critic_value_list

@jax.vmap
def gather(action_probabilities, action_index):
    return action_probabilities[action_index]

@jax.vmap
def actor_loss(log_probability, td_error):
    return -log_probability*td_error

@jax.vmap
def critic_loss(y_true, y_pred):
    # Huber loss
    error          = y_true - y_pred
    is_small_error = jnp.abs(error) < 1
    squared_loss   = jnp.square(error) / 2
    linear_loss    = jnp.abs(error) - 0.5
    #return jnp.where(is_small_error, squared_loss, linear_loss)
    return jnp.square(error)/2.

@jax.jit
def train_a2c_step(optimizer, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - discounted rewards
    # batch[3] - dones
    def loss_fn(model):
        action_probabilities_list, critic_value_list = model(batch[0])
        picked_action_probabilities = gather(action_probabilities_list, batch[1])
        log_probabilities           = jnp.log(picked_action_probabilities)
        td_errors                   = batch[2] - critic_value_list
        actor_losses                = actor_loss(log_probabilities, td_errors)
        critic_losses               = critic_loss(batch[2], critic_value_list)
        return jnp.sum(actor_losses) + jnp.sum(critic_losses)
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss


global_steps    = 0
try:
    for episode in range(num_episodes):
        state           = env.reset()
        episode_rewards = []
        while True:
            global_steps = global_steps+1

            action_probabilities, critic_values  = a2c_inference(a2c_optimizer.target, jnp.asarray([state]))
            print(1, action_probabilities)
            action_probabilities  = np.array(action_probabilities[0])
            print(2, action_probabilities)
            action_probabilities /= action_probabilities.sum()
            print(3, action_probabilities)
            action                = np.random.choice(n_actions, p=action_probabilities)

            new_state, reward, done, _ = env.step(int(action))

            episode_rewards.append(reward)

            if not done:
                replay_memory.append((state, action, reward, new_state, int(done)))

            state = new_state

            if (len(replay_memory)>batch_size):
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                discounted_rewards = np.zeros_like(rewards)
                for t in range(0, batch_size):
                    G_t = 0
                    for idx, j in enumerate(range(t, batch_size)):
                        G_t = G_t + (gamma**idx)*rewards[j]
                    discounted_rewards[t] = G_t
                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-10)
                

                if np.isnan(discounted_rewards).any():
                    print("Damn DISCOUNTED REWARDS!!! HAS A NAN")
                if np.isnan(states).any():
                    print("Damn STATES!!! HAS A NAN")
                if np.isnan(actions).any():
                    print("Damn ACTIONS!!! HAS A NAN")
                if np.isnan(rewards).any():
                    print("Damn REWARDS!!! HAS A NAN")
                if np.isnan(dones).any():
                    print("Damn DONES!!! HAS A NAN")


                a2c_optimizer, loss = train_a2c_step(a2c_optimizer, (
                        jnp.asarray(states),
                        jnp.asarray(actions),
                        jnp.asarray(discounted_rewards),
                        jnp.asarray(dones)
                    ))
                print("Trained...")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break

finally:
    env.close()
