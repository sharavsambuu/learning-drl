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
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99


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
        output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
        return output_dense_layer


env   = gym.make('CartPole-v1')
state = env.reset()

n_actions        = env.action_space.n


actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params)

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model     = flax.nn.Model(critic_module, critic_params)


actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model)


@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def critic_inference(model, x):
    return model(x)

@jax.jit
def backpropagate_critic(optimizer, props):
    # props[0] - state
    # props[1] - next_state
    # props[2] - reward
    # props[3] - done
    def loss_fn(model):
        value      = model(jnp.asarray([props[0]]))[0][0]
        next_value = model(jnp.asarray([props[1]]))[0][0]
        advantage  = props[2]+(gamma*next_value)*(1-props[3]) - value
        return jnp.square(advantage)
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

@jax.jit
def backpropagate_actor(optimizer, critic_model, props):
    # props[0] - state
    # props[1] - next_state
    # props[2] - reward
    # props[3] - done
    # props[4] - action
    value      = critic_model(jnp.asarray([props[0]]))[0][0]
    next_value = critic_model(jnp.asarray([props[1]]))[0][0]
    advantage  = props[2]+(gamma*next_value)*(1-props[3]) - value
    def loss_fn(model, advantage):
        action_probabilities = model(jnp.asarray([props[0]]))[0]
        probability          = action_probabilities[props[4]]
        log_probability      = -jnp.log(probability)
        return log_probability*advantage
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target, advantage)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss


global_step = 0

try:
    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards, dones = [], [], [], []
        while True:
            global_step = global_step+1

            action_probabilities  = actor_inference(actor_optimizer.target, jnp.asarray([state]))
            action_probabilities  = np.array(action_probabilities[0])
            action                = np.random.choice(n_actions, p=action_probabilities)

            next_state, reward, done, _ = env.step(int(action))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(int(done))

            state = next_state

            if debug_render:
                env.render()

            if done:
                print(episode, " - reward :", sum(rewards))

                last_q_value = critic_inference(
                        critic_optimizer.target,
                        jnp.asarray([next_state])
                        )[0][0]

                episode_length = len(rewards)

                q_values       = np.zeros((episode_length, 1))
                for idx, (reward, done) in enumerate(list(zip(rewards, dones))[::-1]):
                    q_values[episode_length-1-idx] = reward + gamma*last_q_value*(1-done)

                values     = critic_inference(critic_optimizer.target, jnp.asarray(states))
                advantages = jnp.subtract(jnp.asarray(q_values), values)


                break
finally:
    env.close()
