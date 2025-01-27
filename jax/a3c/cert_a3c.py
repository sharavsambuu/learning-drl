#
# CERT A3C (Concurrent Experience Replay Trajectories) with A3C
#
#

import os
import random
import math
import time
import threading
import gym
from collections import deque
import flax
import jax
from jax import numpy as jnp
import numpy as np
import queue 


debug_render  = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99
env_name      = "CartPole-v1"
n_workers     = 8


class ActorNetwork(flax.nn.Module): # Actor and Critic networks remain the same (as in previous A3C code)
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
        return output_layer


env       = gym.make(env_name)
state     = env.reset()
n_actions = env.action_space.n
env.close()

actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params) # Global Actor Model (shared)

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model     = flax.nn.Model(critic_module, critic_params) # Global Critic Model (shared)


actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model) # Global Actor Optimizer (shared)
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model) # Global Critic Optimizer (shared)


@jax.jit
def actor_inference(model, x, local_params): # Inference using *local* worker parameters (no change)
    return model.apply({'params': local_params}, x)

@jax.jit
def critic_inference(model, x, local_params): # Inference using *local* worker parameters (no change)
    return model.apply({'params': local_params}, x)

@jax.jit
def backpropagate_critic(optimizer, model, params, props): # Backpropagate Critic using *local* parameters (no change)
    # props[0] - states
    # props[1] - discounted_rewards
    def loss_fn(model):
        values      = model.apply({'params': params}, props[0])[0]
        values      = jnp.reshape(values, (values.shape[0],))
        advantages  = props[1] - values
        return jnp.mean(jnp.square(advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, gradients

@jax.vmap
def gather(probability_vec, action_index): # Gather function remains same (no change)
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(optimizer, critic_model, critic_params, props): # Backpropagate Actor using *local* critic parameters (no change)
    # props[0] - states
    # props[1] - discounted_rewards
    # props[2] - actions
    values      = jax.lax.stop_gradient(critic_inference(critic_model, props[0], critic_params)[0])
    values      = jnp.reshape(values, (values.shape[0],))
    advantages  = props[1] - values
    def loss_fn(model):
        action_probabilities = model.apply({'params': optimizer.target.params}, props[0])[0]
        probabilities        = gather(action_probabilities, props[2])
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, gradients



episode_count = 0
global_step   = 0

gradient_queue = queue.Queue() # Shared queue for gradients (no change)
lock           = threading.Lock() # Lock for global parameter update (no change)


def apply_gradients_thread(): # Separate thread for applying gradients (no change)
    global actor_optimizer
    global critic_optimizer

    while True:
        actor_gradients, critic_gradients = gradient_queue.get()
        if actor_gradients is None:
            break
        with lock:
            actor_optimizer   = actor_optimizer.apply_gradient(actor_gradients)
            critic_optimizer  = critic_optimizer.apply_gradient(critic_gradients)
        gradient_queue.task_done()


gradient_applier_thread = threading.Thread(target=apply_gradients_thread, daemon=True)
gradient_applier_thread.start()


def training_worker(env, thread_index): # Training worker function (emphasizing concurrent trajectories)
    global actor_optimizer
    global critic_optimizer

    global episode_count
    global global_step
    global gradient_queue

    local_actor_params  = actor_optimizer.target.params # Local parameters (no change)
    local_critic_params = critic_optimizer.target.params

    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards, dones = [], [], [], [] # Each worker collects its *own* trajectory

        while True:
            global_step = global_step + 1

            # Inference using *local* parameters - Each worker uses its *own* policy copy
            action_probabilities  = actor_inference(actor_model, jnp.asarray([state]), local_actor_params)
            action_probabilities  = np.array(action_probabilities[0])
            action                = np.random.choice(n_actions, p=action_probabilities)

            next_state, reward, done, _ = env.step(int(action))

            states.append(state) # <<-- Worker's *concurrent trajectory* - collected independently
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state

            if debug_render and thread_index==0:
                env.render()

            if done:
                print("{} step, {} worker, {} episode, reward : {}".format(
                    global_step, thread_index, episode, sum(rewards)))

                episode_length = len(rewards)

                discounted_rewards = np.zeros_like(rewards)
                for t in range(0, episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                    discounted_rewards[t] = G_t
                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-10)


                # Backpropagation using *local* parameters - Worker computes gradients based on its *own* trajectory
                actor_optimizer_local, actor_loss, actor_gradients = backpropagate_actor(
                    actor_optimizer, # Pass global optimizer (for parameter structure)
                    critic_model,
                    local_critic_params, # Use *local* critic parameters for advantage calculation
                    (
                        jnp.asarray(states), # <<-- Worker's *concurrent trajectory* used for backprop
                        jnp.asarray(discounted_rewards),
                        jnp.asarray(actions)
                    )
                )
                critic_optimizer_local, critic_loss, critic_gradients = backpropagate_critic(
                    critic_optimizer, # Pass global optimizer (for parameter structure)
                    critic_model,
                    local_critic_params, # Use *local* critic parameters
                    (
                        jnp.asarray(states), # <<-- Worker's *concurrent trajectory* used for backprop
                        jnp.asarray(discounted_rewards),
                    )
                )

                # Put gradients in the shared queue - Worker contributes gradients *asynchronously*
                gradient_queue.put((actor_gradients, critic_gradients)) # Enqueue gradients for global update

                if episode_count % 1 == 0: # Parameter synchronization (no change)
                    with lock:
                        local_actor_params  = actor_optimizer.target.params
                        local_critic_params = critic_optimizer.target.params


                    episode_count = episode_count + 1

                break
    print("{} id-тэй thread ажиллаж дууслаа.".format(thread_index))


envs = [gym.make(env_name) for i in range(n_workers)]

try:
    workers = [
            threading.Thread(
                target = training_worker,
                daemon = True,
                args   = (envs[i], i)
            ) for i in range(n_workers)
        ]

    for worker in workers:
        time.sleep(1)
        worker.start()

    for worker in workers:
        worker.join()
finally:
    gradient_queue.put(None)
    gradient_applier_thread.join()
    for env in envs: env.close()