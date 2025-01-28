# Soft Actor-Critic, discrete action space
# References:
#  - SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS
#    https://arxiv.org/pdf/1910.07207.pdf
#  - https://github.com/henry-prior/jax-rl
#  - https://github.com/ku2482/sac-discrete.pytorch
#  - https://medium.com/@awjuliani/maximum-entropy-policies-in-reinforcement-learning-everyday-life-f5a1cc18d32d


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
    # props[0] - states
    # props[1] - discounted_rewards
    def loss_fn(model):
        values      = model(props[0])
        values      = jnp.reshape(values,(values.shape[0],))
        advantages  = props[1] - values
        return jnp.mean(jnp.square(advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(optimizer, critic_model, props):
    # props[0] - states
    # props[1] - discounted_rewards
    # props[2] - actions
    values      = jax.lax.stop_gradient(critic_model(props[0]))
    values      = jnp.reshape(values, (values.shape[0],))
    advantages  = jnp.subtract(props[1], values)
    def loss_fn(model):
        action_probabilities      = model(props[0])
        probabilities             = gather(action_probabilities, props[2])
        log_probabilities         = -jnp.log(probabilities)
        alpha                     = 0.4 # Entropy temperature
        entropies                 = -jnp.sum(
                    jnp.multiply(
                        action_probabilities,
                        jnp.log(action_probabilities)
                    ),
                    axis = 1
                )*alpha
        advantages_with_entropies = jnp.add(advantages, entropies)
        return jnp.mean(jnp.multiply(log_probabilities, advantages_with_entropies))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
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

                episode_length = len(rewards)

                discounted_rewards = np.zeros_like(rewards)
                for t in range(0, episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                    discounted_rewards[t] = G_t
                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5) # https://twitter.com/araffin2/status/1329382226421837825

                actor_optimizer, _  = backpropagate_actor(
                    actor_optimizer,
                    critic_optimizer.target,
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                        jnp.asarray(actions)
                    )
                )

                critic_optimizer, _ = backpropagate_critic(
                    critic_optimizer,
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                    )
                )

                break
finally:
    env.close()
