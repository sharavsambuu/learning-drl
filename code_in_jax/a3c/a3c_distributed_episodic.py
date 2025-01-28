# References:
#  - https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/
#  - https://docs.ray.io/en/master/using-ray-with-gpus.html
#  - https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
#      export XLA_PYTHON_CLIENT_ALLOCATOR=platform

import os
import sys
import random
import math
import time
import threading
import multiprocessing
import pickle
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import ray

try:
    ray.init(num_cpus=4, num_gpus=1)
except:
    ray.init(num_cpus=4) # fallback to CPU  if no GPU


debug_render  = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99

env_name      = "CartPole-v1"
n_workers     = 8

env           = gym.make(env_name)
state, info   = env.reset()
state_shape   = state.shape
n_actions     = env.action_space.n
env.close()

class ActorNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        dense_layer_1      = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(dense_layer_1)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=self.n_actions)(activation_layer_2)
        output_layer       = nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        dense_layer_1      = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(dense_layer_1)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=1)(activation_layer_2)
        return output_dense_layer

# Define actor_module and critic_module GLOBALLY
actor_module     = ActorNetwork(n_actions=n_actions)
critic_module    = CriticNetwork()

@jax.jit
def actor_inference(actor_params, state):
    return actor_module.apply({'params': actor_params}, state)

@jax.jit
def critic_inference(critic_params, state):
    return critic_module.apply({'params': critic_params}, state)

@jax.jit
def backpropagate_critic(critic_params, critic_optimizer, states, discounted_rewards):
    def loss_fn(params):
        values      = critic_module.apply({'params': params}, states)
        values      = jnp.reshape(values, (values.shape[0],))
        advantages  = discounted_rewards - values
        return jnp.mean(jnp.square(advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(critic_params)
    updates, critic_optimizer = optax.adam(learning_rate).update(gradients, critic_optimizer, critic_params)
    critic_params = optax.apply_updates(critic_params, updates)
    return critic_params, critic_optimizer, loss

@jax.jit
def backpropagate_actor(actor_params, actor_optimizer, critic_params, states, discounted_rewards, actions):
    values      = jax.lax.stop_gradient(critic_inference(critic_params, states))
    values      = jnp.reshape(values, (values.shape[0],))
    advantages  = discounted_rewards - values
    def loss_fn(params):
        action_probabilities = actor_module.apply({'params': params}, states)
        probabilities        = action_probabilities[jnp.arange(len(actions)), actions]
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(actor_params)
    updates, actor_optimizer = optax.adam(learning_rate).update(gradients, actor_optimizer, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    return actor_params, actor_optimizer, loss

@ray.remote(num_gpus=0.5)
class ParameterServer(object):
    def __init__(self, n_actions, state_shape):
        self.actor_module     = actor_module
        self.critic_module    = critic_module

        dummy_state = jnp.zeros(state_shape)
        self.actor_params  = self.actor_module.init(jax.random.PRNGKey(0), dummy_state)['params']
        self.critic_params = self.critic_module.init(jax.random.PRNGKey(1), dummy_state)['params']

        self.actor_optimizer  = optax.adam(learning_rate).init(self.actor_params)
        self.critic_optimizer = optax.adam(learning_rate).init(self.critic_params)

    def get_actor_params(self,):
        return self.actor_params

    def get_critic_params(self,):
        return self.critic_params

    def update_params(self, actor_params, critic_params):
        self.actor_params  = actor_params
        self.critic_params = critic_params
        self.actor_optimizer  = optax.adam(learning_rate).init(self.actor_params)
        self.critic_optimizer = optax.adam(learning_rate).init(self.critic_params)

@ray.remote(num_gpus=0.04)
class RolloutWorker(object):
    def __init__(self, worker_index, n_actions):
        self.worker_index = worker_index
        self.n_actions = n_actions
        self.env = gym.make(env_name, render_mode='human' if debug_render and worker_index==0 else None)

    def rollout_episode(self, actor_params, critic_params):
        state, info = self.env.reset()
        states, actions, rewards, dones = [], [], [], []
        while True:
            action_probabilities = actor_inference(actor_params, jnp.asarray(state))
            action_probabilities = np.array(action_probabilities)
            action = np.random.choice(self.n_actions, p=action_probabilities)

            next_state, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(int(done))

            state = next_state

            if done:
                print("worker {} episode, reward : {}".format(self.worker_index, sum(rewards)))
                episode_length = len(rewards)
                discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
                for t in range(episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t += (gamma**idx) * rewards[j] * (1 - dones[j])
                    discounted_rewards[t] = G_t
                discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

                actor_batch = (states, discounted_rewards, actions)
                critic_batch = (states, discounted_rewards)
                return actor_batch, critic_batch

@ray.remote(num_cpus=1)
def learner(parameter_server, actor_batches, critic_batches):
    actor_params = ray.get(parameter_server.get_actor_params.remote())
    critic_params = ray.get(parameter_server.get_critic_params.remote())
    actor_optimizer = optax.adam(learning_rate).init(actor_params)
    critic_optimizer = optax.adam(learning_rate).init(critic_params)
    for i in range(len(actor_batches)):
        actor_params, actor_optimizer, _ = backpropagate_actor(
            actor_params,
            actor_optimizer,
            critic_params,
            jnp.asarray(actor_batches[i][0]),
            jnp.asarray(actor_batches[i][1]),
            jnp.asarray(actor_batches[i][2], dtype=jnp.int32)
        )
    for i in range(len(critic_batches)):
        critic_params, critic_optimizer, _ = backpropagate_critic(
            critic_params,
            critic_optimizer,
            jnp.asarray(critic_batches[i][0]),
            jnp.asarray(critic_batches[i][1]),
        )
    parameter_server.update_params.remote(actor_params, critic_params)

if __name__ == "__main__":
    parameter_server = ParameterServer.remote(n_actions, state_shape)
    workers = [RolloutWorker.remote(worker_index=i, n_actions=n_actions) for i in range(n_workers)]

    futures = []
    for _ in range(n_workers):
        actor_params = ray.get(parameter_server.get_actor_params.remote())
        critic_params = ray.get(parameter_server.get_critic_params.remote())
        worker = random.choice(workers)
        futures.append(worker.rollout_episode.remote(actor_params, critic_params))

    for i in range(0, num_episodes, n_workers):
        ready_futures, futures = ray.wait(futures, num_returns=n_workers)

        actor_batches = []
        critic_batches = []
        for future in ready_futures:
            actor_batch, critic_batch = ray.get(future)
            actor_batches.append(actor_batch)
            critic_batches.append(critic_batch)

        learner.remote(parameter_server, actor_batches, critic_batches)

        for _ in range(n_workers):
            actor_params = ray.get(parameter_server.get_actor_params.remote())
            critic_params = ray.get(parameter_server.get_critic_params.remote())
            worker = random.choice(workers)
            futures.append(worker.rollout_episode.remote(actor_params, critic_params))

    ray.shutdown()