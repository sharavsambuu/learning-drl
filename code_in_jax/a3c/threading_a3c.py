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
import queue # For gradient queue


debug_render  = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99
env_name      = "CartPole-v1"
n_workers     = 8


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


env       = gym.make(env_name)
state     = env.reset()
n_actions = env.action_space.n
env.close()

actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params) # Global Actor Model

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model     = flax.nn.Model(critic_module, critic_params) # Global Critic Model


actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model ) # Global Actor Optimizer
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model) # Global Critic Optimizer


@jax.jit
def actor_inference(model, x, local_params): # Inference using *local* worker parameters
    return model.apply({'params': local_params}, x)

@jax.jit
def critic_inference(model, x, local_params): # Inference using *local* worker parameters
    return model.apply({'params': local_params}, x)

@jax.jit
def backpropagate_critic(optimizer, model, params, props): # Backpropagate using *local* parameters
    # props[0] - states
    # props[1] - discounted_rewards
    def loss_fn(model):
        values      = model.apply({'params': params}, props[0])[0] # Use provided parameters for loss calculation
        values      = jnp.reshape(values, (values.shape[0],))
        advantages  = props[1] - values
        return jnp.mean(jnp.square(advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, gradients # Return gradients as well

@jax.vmap
def gather(probability_vec, action_index): 
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(optimizer, critic_model, critic_params, props): # Backpropagate Actor using *local* critic parameters
    # props[0] - states
    # props[1] - discounted_rewards
    # props[2] - actions
    values      = jax.lax.stop_gradient(critic_inference(critic_model, props[0], critic_params)[0]) # Use provided *local* critic parameters
    values      = jnp.reshape(values, (values.shape[0],))
    advantages  = props[1] - values
    def loss_fn(model):
        action_probabilities = model.apply({'params': optimizer.target.params}, props[0])[0] # Use *local* actor parameters
        probabilities        = gather(action_probabilities, props[2])
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, gradients # Return gradients as well


episode_count = 0
global_step   = 0

gradient_queue = queue.Queue() # Shared queue for gradients
lock           = threading.Lock() # Lock for global parameter update


def apply_gradients_thread(): # Separate thread for applying gradients
    global actor_optimizer
    global critic_optimizer

    while True:
        actor_gradients, critic_gradients = gradient_queue.get() # Get gradients from queue
        if actor_gradients is None: # Sentinel value to stop thread
            break
        with lock: # Lock only for applying gradients to global optimizers
            actor_optimizer   = actor_optimizer.apply_gradient(actor_gradients)   # Apply accumulated actor gradients
            critic_optimizer  = critic_optimizer.apply_gradient(critic_gradients) # Apply accumulated critic gradients
        gradient_queue.task_done() # Indicate task completion


gradient_applier_thread = threading.Thread(target=apply_gradients_thread, daemon=True) # Create gradient applier thread
gradient_applier_thread.start() # Start gradient applier thread


def training_worker(env, thread_index): # Training worker function (modified for asynchronous updates)
    global actor_optimizer
    global critic_optimizer

    global episode_count
    global global_step
    global gradient_queue

    # Local copies of global model parameters for each worker - Workers operate on *local* parameters for inference and backpropagation
    local_actor_params  = actor_optimizer.target.params
    local_critic_params = critic_optimizer.target.params


    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards, dones = [], [], [], []

        while True:
            global_step = global_step + 1

            # Inference using *local* parameters
            action_probabilities  = actor_inference(actor_model, jnp.asarray([state]), local_actor_params) # Use local actor parameters for inference
            action_probabilities  = np.array(action_probabilities[0])
            action                = np.random.choice(n_actions, p=action_probabilities)

            next_state, reward, done, _ = env.step(int(action))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(int(done))

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


                # Backpropagation using *local* parameters - Workers compute gradients *independently*
                actor_optimizer_local, actor_loss, actor_gradients = backpropagate_actor( # Backpropagate Actor - get gradients
                    actor_optimizer, # Pass global optimizer (for parameter structure), but gradients are computed locally
                    critic_model,
                    local_critic_params, # Use *local* critic parameters
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                        jnp.asarray(actions)
                    )
                )
                critic_optimizer_local, critic_loss, critic_gradients = backpropagate_critic( # Backpropagate Critic - get gradients
                    critic_optimizer, # Pass global optimizer (for parameter structure), but gradients are computed locally
                    critic_model,
                    local_critic_params, # Use *local* critic parameters
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                    )
                )

                # Put gradients in the shared queue - Workers *asynchronously* enqueue gradients
                gradient_queue.put((actor_gradients, critic_gradients)) # Enqueue gradients for global application

                # Update local worker parameters with the *global* parameters after gradient update (less frequent sync)
                if episode_count % 1 == 0: # Sync local parameters with global parameters every episode (adjust frequency as needed)
                    with lock: # Lock when accessing global parameters
                        local_actor_params  = actor_optimizer.target.params # Get latest global actor parameters
                        local_critic_params = critic_optimizer.target.params # Get latest global critic parameters


                    episode_count = episode_count + 1 # Increment episode count outside lock

                break # Episode done
    print("{} id-тэй thread ажиллаж дууслаа.".format(thread_index)) # Worker thread finished


envs = [gym.make(env_name) for i in range(n_workers)] # Create multiple environments for workers

try:
    workers = [ # Create worker threads
            threading.Thread(
                target = training_worker,
                daemon = True,
                args   = (envs[i], i)
            ) for i in range(n_workers)
        ]

    for worker in workers: # Start worker threads
        time.sleep(1)
        worker.start()

    for worker in workers: # Wait for all workers to finish (which won't happen in daemon threads)
        worker.join()
finally:
    gradient_queue.put(None) # Send sentinel value to stop gradient applier thread
    gradient_applier_thread.join() # Wait for gradient applier thread to finish
    for env in envs: env.close() # Close environments