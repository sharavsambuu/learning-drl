import os
import random
import math
import gym

import flax
import jax
from jax import numpy as jnp
import numpy as np
import numpy

numpy.set_printoptions(precision=15)

debug_render  = True
debug         = False
num_episodes  = 600
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


env   = gym.make('CartPole-v1')
state = env.reset()

n_actions        = env.action_space.n

pg_module        = PolicyNetwork.partial(n_actions=n_actions)
_, params        = pg_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
policy_network   = flax.nn.Model(pg_module, params)

optimizer        = flax.optim.Adam(learning_rate).create(policy_network)

@jax.jit
def policy_inference(model, x):
    action_probabilities = model(x)
    return action_probabilities

@jax.vmap
def gather(action_probabilities, action_index):
    return action_probabilities[action_index]

@jax.jit
def train_step(optimizer, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - discounted rewards
    def loss_fn(model):
        action_probabilities_list   = model(batch[0])
        picked_action_probabilities = gather(action_probabilities_list, batch[1])
        log_probabilities           = jnp.log(picked_action_probabilities)
        losses                      = jnp.multiply(log_probabilities, batch[2])
        return -jnp.sum(losses)
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

global_steps = 0
try:
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        while True:
            global_steps = global_steps+1

            action_probabilities  = policy_inference(optimizer.target, jnp.asarray([state]))[0]
            action_probabilities  = np.array(action_probabilities)
            action_probabilities /= action_probabilities.sum()
            action                = np.random.choice(n_actions, p=action_probabilities)

            new_state, reward, done, _ = env.step(int(action))

            states.append(state)
            actions.append(action)
            rewards.append(reward)

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
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5) # https://twitter.com/araffin2/status/1329382226421837825

                print("Training...")
                optimizer, loss = train_step(optimizer, (
                        jnp.asarray(states),
                        jnp.asarray(actions),
                        jnp.asarray(discounted_rewards)
                    ))

                break
finally:
    env.close()
