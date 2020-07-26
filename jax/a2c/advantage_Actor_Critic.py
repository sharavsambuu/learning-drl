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


# References:
#    - https://github.com/google/flax/blob/master/examples/vae/train.py

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
    action_probabilities, critic_value = model(x)
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
    # Temporal Difference => (reward + gamma*V') - V 
    td_errors       = ((optimizer.target(batch[2])*gamma)+batch[1])-optimizer.target(batch[0])
    return optimizer, td_errors


global_steps = 0
try:
    for episode in range(num_episodes):
        states, actions, rewards, next_states = [], [], [], []
        state = env.reset()
        while True:
            global_steps = global_steps+1

            action_probabilities  = actor_inference(a2c_optimizer.target, jnp.asarray([state]))[0]
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

                critic_optimizer, td_errors = train_critic_step(critic_optimizer, (
                        jnp.asarray(states),
                        jnp.asarray(rewards),
                        jnp.asarray(next_states)
                    ))

                actor_optimizer, loss = train_actor_step(actor_optimizer, (
                        jnp.asarray(states),
                        jnp.asarray(actions),
                        jnp.asarray(td_errors)
                    ))

                break
finally:
    env.close()
