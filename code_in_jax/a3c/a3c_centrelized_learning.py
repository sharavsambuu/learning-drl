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
    ray.init()

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

# No need for separate gather function in JAX, direct indexing works efficiently
@jax.jit
def backpropagate_actor(actor_params, actor_optimizer, critic_params, states, discounted_rewards, actions):
    values      = jax.lax.stop_gradient(critic_inference(critic_params, states))
    values      = jnp.reshape(values, (values.shape[0],))
    advantages  = discounted_rewards - values
    def loss_fn(params):
        action_probabilities = actor_module.apply({'params': params}, states)
        probabilities        = action_probabilities[jnp.arange(len(actions)), actions] # Efficiently gather probabilities
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(actor_params)
    updates, actor_optimizer = optax.adam(learning_rate).update(gradients, actor_optimizer, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    return actor_params, actor_optimizer, loss

@ray.remote(num_gpus=0.5)
class BrainServer(object):
    def __init__(self, n_actions, state_shape):
        self.actor_module     = actor_module # Use global actor_module
        self.critic_module    = critic_module # Use global critic_module

        dummy_state = jnp.zeros(state_shape)
        self.actor_params  = self.actor_module.init(jax.random.PRNGKey(0), dummy_state)['params']
        self.critic_params = self.critic_module.init(jax.random.PRNGKey(1), dummy_state)['params']

        self.actor_optimizer  = optax.adam(learning_rate).init(self.actor_params)
        self.critic_optimizer = optax.adam(learning_rate).init(self.critic_params)

    def act(self, state):
        action_probabilities = actor_inference(self.actor_params, state)
        action_probabilities = np.array(action_probabilities)
        action               = np.random.choice(n_actions, p=action_probabilities)
        return action

    def learn(self, actor_batch, critic_batch):
        # Corrected line: Now unpack 3 values, but ignore loss
        self.actor_params, self.actor_optimizer, _ = backpropagate_actor(
            self.actor_params,
            self.actor_optimizer,
            self.critic_params,
            jnp.asarray(np.array(actor_batch[0])),
            jnp.asarray(np.array(actor_batch[1])),
            jnp.asarray(np.array(actor_batch[2]), dtype=jnp.int32)
        )
        # Corrected line: Now unpack 3 values, but ignore loss
        self.critic_params, self.critic_optimizer, _ = backpropagate_critic(
            self.critic_params,
            self.critic_optimizer,
            jnp.asarray(np.array(critic_batch[0])),
            jnp.asarray(np.array(critic_batch[1])),
        )

    def get_actor_params(self):
        return self.actor_params



@ray.remote(num_gpus=0.04)
class RolloutWorker(object):
    def __init__(self, worker_index, brain_server_handle):
        self.worker_index = worker_index
        self.brain_server_handle = brain_server_handle
        self.env = gym.make(env_name, render_mode='human' if debug_render and worker_index==0 else None)
        self.n_actions = n_actions

    def rollout_episodes(self) -> tuple[tuple, tuple]:
        actor_batch, critic_batch = None, None

        for episode in range(num_episodes):
            state, info = self.env.reset()
            states, actions, rewards, dones = [], [], [], []

            while True:
                actor_params = ray.get(self.brain_server_handle.get_actor_params.remote())
                action_probabilities = actor_inference(actor_params, jnp.asarray(state))
                action_probabilities = np.array(action_probabilities)
                action               = np.random.choice(self.n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = self.env.step(int(action))
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(int(done))

                state = next_state

                if done:
                    print("worker {} episode {}, reward : {}".format(self.worker_index, episode, sum(rewards)))
                    episode_length     = len(rewards)
                    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
                    for t in range(episode_length):
                        G_t = 0
                        for idx, j in enumerate(range(t, episode_length)):
                            G_t += (gamma**idx)*rewards[j]*(1-dones[j])
                        discounted_rewards[t] = G_t
                    discounted_rewards  = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards)+1e-8)

                    actor_batch  = (states, discounted_rewards, actions)
                    critic_batch = (states, discounted_rewards)
                    break

            return actor_batch, critic_batch



if __name__ == "__main__":
    brain_server = BrainServer.remote(n_actions, state_shape)

    workers = [RolloutWorker.remote(worker_index=i, brain_server_handle=brain_server) for i in range(n_workers)]

    actor_batch_futures_list = []
    critic_batch_futures_list = []

    for worker in workers:
        future = worker.rollout_episodes.remote()
        actor_batch_futures_list.append(future)
        critic_batch_futures_list.append(future)

    num_episodes_per_iteration = num_episodes // 10
    if num_episodes_per_iteration < 1: num_episodes_per_iteration = 1

    worker_actor_handles = list(workers)

    for iteration in range(10):
        start_time = time.time()
        for _ in range(num_episodes_per_iteration):

            actor_batches_list_iteration  = []
            critic_batches_list_iteration = []

            ready_ids, actor_batch_futures_list   = ray.wait(actor_batch_futures_list)
            ready_ids_critic, critic_batch_futures_list = ray.wait(critic_batch_futures_list)

            for ready_future in ready_ids:
                actor_batch, critic_batch = ray.get(ready_future)
                actor_batches_list_iteration.append(actor_batch)
                critic_batches_list_iteration.append(critic_batch)
                worker_index_to_restart = ready_ids.index(ready_future)
                worker_to_restart = worker_actor_handles[worker_index_to_restart]
                actor_batch_futures_list.append(worker_to_restart.rollout_episodes.remote())
                critic_batch_futures_list.append(worker_to_restart.rollout_episodes.remote())

            aggregated_actor_states     = []
            aggregated_actor_rewards    = []
            aggregated_actor_actions    = []
            aggregated_critic_states    = []
            aggregated_critic_rewards   = []

            for actor_batch in actor_batches_list_iteration:
                aggregated_actor_states.extend(actor_batch[0])
                aggregated_actor_rewards.extend(actor_batch[1])
                aggregated_actor_actions.extend(actor_batch[2])
            actor_learn_batch = (aggregated_actor_states, aggregated_actor_rewards, aggregated_actor_actions)

            for critic_batch in critic_batches_list_iteration:
                aggregated_critic_states.extend(critic_batch[0])
                aggregated_critic_rewards.extend(critic_batch[1])
            critic_learn_batch = (aggregated_critic_states, aggregated_critic_rewards)

            brain_server.learn.remote(actor_learn_batch, critic_learn_batch)

        elapsed_time = time.time() - start_time
        print("Iteration {} completed in {:.2f} seconds. Episodes per iteration: {}".format(iteration, elapsed_time, num_episodes_per_iteration))

    ray.shutdown()