# References:
#  - https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/

import os
import sys
import random
import math
import time
import threading
import multiprocessing
import pickle
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np
import uuid
import ray; ray.init(num_cpus=4)


debug_render  = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99

env_name      = "CartPole-v1"
n_workers     = 8

env           = gym.make(env_name)
state_shape   = env.reset().shape
n_actions     = env.action_space.n
env.close()

@jax.jit
def act_local(model, state):
    return model(jnp.asarray([state]))[0]

@ray.remote
def rollout_worker(brain_server):
    class ActorNetwork(flax.nn.Module):
        def apply(self_, x, n_actions):
            dense_layer_1      = flax.nn.Dense(x, 64)
            activation_layer_1 = flax.nn.relu(dense_layer_1)
            dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
            activation_layer_2 = flax.nn.relu(dense_layer_2)
            output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
            output_layer       = flax.nn.softmax(output_dense_layer)
            return output_layer

    actor_module      = ActorNetwork.partial(n_actions=n_actions)
    _, actor_params   = actor_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])

    local_actor_model = flax.nn.Model(actor_module, actor_params)

    env       = gym.make(env_name)

    worker_id = str(uuid.uuid4())
    try:
        for episode in range(num_episodes):
            state = env.reset()
            states, actions, rewards, dones = [], [], [], []

            while True:
                action_probabilities = act_local(local_actor_model, state)
                action_probabilities = np.array(action_probabilities)
                action               = np.random.choice(n_actions, p=action_probabilities)

                next_state, reward, done, _ = env.step(int(action))

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(int(done))

                state = next_state

                if done:
                    print("worker {}, episode {}, reward : {}".format(worker_id, episode, sum(rewards)))
                    episode_length     = len(rewards)
                    discounted_rewards = np.zeros_like(rewards)
                    for t in range(0, episode_length):
                        G_t = 0
                        for idx, j in enumerate(range(t, episode_length)):
                            G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                        discounted_rewards[t] = G_t
                    discounted_rewards  = discounted_rewards - np.mean(discounted_rewards)
                    discounted_rewards  = discounted_rewards / (np.std(discounted_rewards)+1e-10)

                    new_actor_weights = ray.get(brain_server.learn.remote(
                                ( states, discounted_rewards,actions),
                                ( states, discounted_rewards)
                            ))
                    local_actor_model = local_actor_model.replace(params=new_actor_weights)
                    break
    finally:
        env.close()
        return 0
    return 0


@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def backpropagate_critic(optimizer, props):
    # props[0] - states
    # props[1] - discounted_rewards
    def loss_fn(model):
        values      = model(props[0])
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
    advantages  = props[1] - values
    def loss_fn(model):
        action_probabilities = model(props[0])
        probabilities        = gather(action_probabilities, props[2])
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

@ray.remote
class BrainServer(object):
    def __init__(self,):
        class ActorNetwork(flax.nn.Module):
            def apply(self_, x, n_actions):
                dense_layer_1      = flax.nn.Dense(x, 64)
                activation_layer_1 = flax.nn.relu(dense_layer_1)
                dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
                activation_layer_2 = flax.nn.relu(dense_layer_2)
                output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
                output_layer       = flax.nn.softmax(output_dense_layer)
                return output_layer
        class CriticNetwork(flax.nn.Module):
            def apply(self_, x):
                dense_layer_1      = flax.nn.Dense(x, 64)
                activation_layer_1 = flax.nn.relu(dense_layer_1)
                dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
                activation_layer_2 = flax.nn.relu(dense_layer_2)
                output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
                return output_dense_layer

        self.actor_module     = ActorNetwork.partial(n_actions=n_actions)
        _, self.actor_params  = self.actor_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])
        self.actor_model      = flax.nn.Model(self.actor_module, self.actor_params)

        self.critic_module    = CriticNetwork.partial()
        _, self.critic_params = self.critic_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])
        self.critic_model     = flax.nn.Model(self.critic_module, self.critic_params)

        self.actor_optimizer  = flax.optim.Adam(learning_rate).create(self.actor_model)
        self.critic_optimizer = flax.optim.Adam(learning_rate).create(self.critic_model)
        pass

    def act(self, state):
        action_probabilities = actor_inference(self.actor_optimizer.target, jnp.asarray([state]))
        action_probabilities = np.array(action_probabilities[0])
        action               = np.random.choice(n_actions, p=action_probabilities)
        return action

    def learn(self, actor_batch, critic_batch):
        self.actor_optimizer, _ = backpropagate_actor(
            self.actor_optimizer,
            self.critic_optimizer.target,
            (
                jnp.asarray(actor_batch[0]), # states
                jnp.asarray(actor_batch[1]), # discounted_rewards
                jnp.asarray(actor_batch[2])  # actions
            )
        )
        self.critic_optimizer, _ = backpropagate_critic(
            self.critic_optimizer,
            (
                jnp.asarray(critic_batch[0]), # states
                jnp.asarray(critic_batch[1]), # discounted_rewards
            )
        )
        return self.actor_optimizer.target.params


brain_server = BrainServer.remote()

result_ids = [rollout_worker.remote(brain_server) for _ in range(n_workers)]

while len(result_ids):
    done_id, result_ids = ray.wait(result_ids)
    print(done_id, "id-тай даалгавар ажиллаж дууслаа.")

