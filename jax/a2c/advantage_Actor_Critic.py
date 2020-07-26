import os
import random
import math
import gym

import flax
import jax
from jax import numpy as jnp
import numpy as np
import numpy


debug_render  = True
debug         = False
num_episodes  = 600
learning_rate = 0.001
gamma         = 0.99 # discount factor


class ActorNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(flax.nn.Module):
    def apply(self, x):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, 1)
        return output_layer


env   = gym.make('CartPole-v1')
state = env.reset()

n_actions        = env.action_space.n

actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_network    = flax.nn.Model(actor_module, actor_params)

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_network   = flax.nn.Model(critic_module, critic_params)

actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_network)
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_network)


@jax.jit
def actor_inference(model, x):
    action_probabilities = model(x)
    return action_probabilities

@jax.vmap
def gather(action_probabilities, action_index):
    return action_probabilities[action_index]

@jax.jit
def train_actor_step(optimizer, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - advantages
    def loss_fn(model):
        action_probabilities_list   = model(batch[0])
        picked_action_probabilities = gather(action_probabilities_list, batch[1])
        log_probabilities           = jnp.log(picked_action_probabilities)
        losses                      = jnp.multiply(log_probabilities, batch[2])
        return -jnp.sum(losses)
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss


@jax.vmap
def critic_loss(value, next_value, reward):
    td_target = reward + gamma*next_value
    td_error  = jax.lax.stop_gradient(td_target)-value
    return jnp.square(td_error)

@jax.jit
def train_critic_step(optimizer, batch):
    # batch[0] - states
    # batch[1] - rewards
    # batch[2] - next states
    def loss_fn(model):
        values      = model(batch[0])
        next_values = model(batch[2])
        losses      = critic_loss(values, next_values, batch[1])
        return jnp.mean(losses)
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    td_errors       = optimizer.target(batch[2])-optimizer.target(batch[0])
    return optimizer, jnp.absolute(td_errors)


global_steps = 0
try:
    for episode in range(num_episodes):
        states, actions, rewards, next_states = [], [], [], []
        state = env.reset()
        while True:
            global_steps = global_steps+1

            action_probabilities  = actor_inference(actor_optimizer.target, jnp.asarray([state]))[0]
            action_probabilities  = np.array(action_probabilities)
            action_probabilities /= action_probabilities.sum()
            action                = np.random.choice(n_actions, p=action_probabilities)

            new_state, reward, done, _ = env.step(int(action))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(new_state)

            state = new_state

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(rewards)))

                episode_length     = len(rewards)
                discounted_rewards = np.zeros_like(rewards)
                for t in range(0, episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t = G_t + (gamma**idx)*rewards[j]
                    discounted_rewards[t] = G_t
                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-10)

                critic_optimizer, _ = train_critic_step(critic_optimizer, (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                        jnp.asarray(next_states)
                    ))

                td_errors = discounted_rewards-critic_optimizer.target(jnp.asarray(states))

                actor_optimizer, loss = train_actor_step(actor_optimizer, (
                        jnp.asarray(states),
                        jnp.asarray(actions),
                        jnp.asarray(td_errors)
                    ))

                break
finally:
    env.close()
