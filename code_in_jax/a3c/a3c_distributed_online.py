# References:
#  - https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/
#  - https://docs.ray.io/en/master/ray-with-gpus.html
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
    ray.init() # fallback to CPU if no GPU

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
def backpropagate_critic(critic_params, critic_optimizer, state, next_state, reward, done):
    next_value = jax.lax.stop_gradient(critic_inference(critic_params, next_state))
    target_value = reward + gamma * next_value * (1 - done)

    def loss_fn(params):
        current_value = critic_module.apply({'params': params}, state)
        loss = jnp.mean(jnp.square(target_value - current_value))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(critic_params)
    critic_optimizer = critic_optimizer.update(grads, critic_optimizer, critic_params)
    critic_params = optax.apply_updates(critic_params, critic_optimizer.state.updates)
    return critic_params, critic_optimizer, loss

@jax.jit
def backpropagate_actor(actor_params, actor_optimizer, critic_params, state, action, advantage):
    def loss_fn(params):
        action_probabilities = actor_module.apply({'params': params}, state)
        log_probability = jnp.log(action_probabilities[0, action])
        loss = -log_probability * advantage
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(actor_params)
    actor_optimizer = actor_optimizer.update(grads, actor_optimizer, actor_params)
    actor_params = optax.apply_updates(actor_params, actor_optimizer.state.updates)
    return actor_params, actor_optimizer, loss

@ray.remote(num_gpus=0.5)
class ParameterServer(object):
    def __init__(self, n_actions, state_shape):
        self.actor_module = actor_module
        self.critic_module = critic_module

        dummy_state = jnp.zeros(state_shape)
        self.actor_params = self.actor_module.init(jax.random.PRNGKey(0), dummy_state)['params']
        self.critic_params = self.critic_module.init(jax.random.PRNGKey(1), dummy_state)['params']

        self.actor_optimizer = optax.adam(learning_rate).init(self.actor_params)
        self.critic_optimizer = optax.adam(learning_rate).init(self.critic_params)

    def get_actor_params(self):
        return self.actor_params, self.actor_optimizer.state

    def get_critic_params(self):
        return self.critic_params, self.critic_optimizer.state

    def update_params(self, actor_params, critic_params, actor_optimizer_state, critic_optimizer_state):
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.actor_optimizer = self.actor_optimizer.replace(state=actor_optimizer_state)
        self.critic_optimizer = self.critic_optimizer.replace(state=critic_optimizer_state)

@ray.remote(num_gpus=0.04)
class RolloutWorker(object):
    def __init__(self, worker_index, n_actions):
        self.worker_index = worker_index
        self.n_actions = n_actions
        self.env = gym.make(env_name, render_mode='human' if debug_render and worker_index == 0 else None)

    def rollout_episode(self, parameter_server):
        for episode in range(num_episodes):
            state, info = self.env.reset()
            states, actions, rewards, dones = [], [], [], []
            actor_optimizer_state_worker = None
            critic_optimizer_state_worker = None
            
            step = 0
            while True:
                (actor_params_server, actor_optimizer_state_server), (critic_params_server, critic_optimizer_state_server) = ray.get(parameter_server.get.remote(['actor_params', 'critic_params']))
                #print("worker {} received".format(self.worker_index))

                if actor_optimizer_state_worker is None:
                    actor_optimizer_state_worker = actor_optimizer_state_server
                    critic_optimizer_state_worker = critic_optimizer_state_server

                action_probabilities = actor_inference(actor_params_server, jnp.asarray([state]))
                action_probabilities = np.array(action_probabilities[0])
                action = np.random.choice(self.n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = self.env.step(int(action))
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(int(done))

                critic_params_server, critic_optimizer_state_worker, _ = backpropagate_critic(
                    critic_params_server,
                    optax.adam(learning_rate).init(critic_params_server).replace(state=critic_optimizer_state_worker),
                    jnp.asarray([state]),
                    jnp.asarray([next_state]),
                    reward,
                    int(done)
                )

                value = critic_inference(critic_params_server, jnp.asarray([state]))[0][0]
                next_value = critic_inference(critic_params_server, jnp.asarray([next_state]))[0][0]
                advantage = reward + (gamma * next_value) * (1 - int(done)) - value

                actor_params_server, actor_optimizer_state_worker, _ = backpropagate_actor(
                    actor_params_server,
                    optax.adam(learning_rate).init(actor_params_server).replace(state=actor_optimizer_state_worker),
                    critic_params_server,
                    jnp.asarray([state]),
                    action,
                    advantage
                )

                print("worker {} updated".format(self.worker_index))
                parameter_server.update_params.remote(actor_params_server, critic_params_server, actor_optimizer_state_worker, critic_optimizer_state_worker)

                state = next_state
                step += 1

                if done:
                    print("worker {} episode {}, reward : {}".format(self.worker_index, episode, sum(rewards)))
                    break
        return 0

if __name__ == "__main__":
    parameter_server = ParameterServer.remote(n_actions, state_shape)
    workers = [RolloutWorker.remote(worker_index=i, n_actions=n_actions) for i in range(n_workers)]

    futures = [worker.rollout_episode.remote(parameter_server) for worker in workers]

    ray.wait(futures, num_returns=len(futures))

    ray.shutdown()