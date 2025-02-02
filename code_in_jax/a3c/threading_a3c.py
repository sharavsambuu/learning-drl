import os
import random
import math
import time
import threading
import gymnasium as gym
from collections import deque

import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import queue 


debug_render  = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99
env_name      = "CartPole-v1"
n_workers     = 8


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


env         = gym.make(env_name)
state, info = env.reset()
n_actions   = env.action_space.n
env.close()

actor_module            = ActorNetwork(n_actions=n_actions)
critic_module           = CriticNetwork()
dummy_state             = jnp.zeros(state.shape)
global_actor_params     = actor_module.init(jax.random.PRNGKey(0), dummy_state)['params']
global_critic_params    = critic_module.init(jax.random.PRNGKey(1), dummy_state)['params']
global_actor_optimizer  = optax.adam(learning_rate).init(global_actor_params)
global_critic_optimizer = optax.adam(learning_rate).init(global_critic_params)


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
    return gradients # Return gradients

@jax.jit
def backpropagate_actor(actor_params, critic_params, states, discounted_rewards, actions): 
    values      = jax.lax.stop_gradient(critic_inference(critic_params, states))
    values      = jnp.reshape(values, (values.shape[0],))
    advantages  = discounted_rewards - values
    def loss_fn(params):
        action_probabilities = actor_module.apply({'params': params}, states)
        probabilities        = action_probabilities[jnp.arange(len(actions)), actions] 
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(actor_params)
    return gradients 


episode_count     = 0
global_step       = 0
gradient_queue    = queue.Queue()     # Shared queue for gradients
lock              = threading.Lock()  # Lock for global parameter update
training_finished = False             # Global flag for Ctrl+C


def apply_gradients_thread(): # Separate thread for applying gradients
    global global_actor_optimizer
    global global_critic_optimizer
    global gradient_queue
    global training_finished

    try:
        while not training_finished: 
            actor_gradients, critic_gradients = gradient_queue.get(timeout=1) 
            if actor_gradients is None: # Sentinel value to stop thread
                break
            with lock: # Lock only for applying gradients to global optimizers
                updates_actor, global_actor_optimizer = optax.adam(learning_rate).update(actor_gradients, global_actor_optimizer, global_actor_params)
                global_actor_params = optax.apply_updates(global_actor_params, updates_actor)
                updates_critic, global_critic_optimizer = optax.adam(learning_rate).update(critic_gradients, global_critic_optimizer, global_critic_params)
                global_critic_params = optax.apply_updates(global_critic_params, updates_critic)
            gradient_queue.task_done() # Indicate task completion
    except queue.Empty: # Handle timeout exception if queue.get times out (for graceful exit on Ctrl+C)
        print("Gradient applier thread timeout, exiting...")
        pass
    except KeyboardInterrupt: # Handle KeyboardInterrupt in gradient applier thread
        print("Gradient applier thread received KeyboardInterrupt, exiting...")
        pass
    finally:
        print("Gradient applier thread finished.")


gradient_applier_thread = threading.Thread(target=apply_gradients_thread, daemon=True) 
gradient_applier_thread.start() 


def training_worker(env, thread_index): 
    global global_actor_params
    global global_critic_params
    global gradient_queue
    global episode_count
    global global_step
    global training_finished

    local_actor_params  = global_actor_params 
    local_critic_params = global_critic_params

    try: 
        for episode in range(num_episodes):
            if training_finished: 
                break

            state, info = env.reset()
            states, actions, rewards, dones = [], [], [], []

            while True:
                global_step = global_step + 1
                if training_finished: 
                    break

                action_probabilities  = actor_inference(local_actor_params, jnp.asarray(state)) 
                action_probabilities  = np.array(action_probabilities)
                action                = np.random.choice(n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                states .append(state    )
                actions.append(action   )
                rewards.append(reward   )
                dones  .append(int(done))

                state = next_state

                if debug_render and thread_index==0:
                    env.render()

                if done:
                    print("{} step, {} worker, {} episode, reward : {}".format(
                        global_step, thread_index, episode, sum(rewards)))

                    episode_length = len(rewards)

                    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
                    for t in range(episode_length):
                        G_t = 0
                        for idx, j in enumerate(range(t, episode_length)):
                            G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                        discounted_rewards[t] = G_t
                    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards)+1e-8)


                    actor_gradients = backpropagate_actor( 
                        local_actor_params, 
                        local_critic_params, 
                        jnp.asarray(np.array(states)),
                        jnp.asarray(np.array(discounted_rewards)),
                        jnp.asarray(np.array(actions), dtype=jnp.int32)
                    )
                    critic_gradients = backpropagate_critic( 
                        local_critic_params, 
                        global_critic_optimizer, 
                        jnp.asarray(np.array(states)),
                        jnp.asarray(np.array(discounted_rewards)),
                    )

                    gradient_queue.put((actor_gradients, critic_gradients)) 

                    if episode_count % 1 == 0: 
                        with lock: 
                            local_actor_params  = global_actor_params 
                            local_critic_params = global_critic_params 
                    episode_count += 1
                    break 
        print("{} id-тэй thread ажиллаж дууслаа.".format(thread_index)) 
    except KeyboardInterrupt: 
        print("{} id-тэй thread-д KeyboardInterrupt exception!".format(thread_index))
        pass 


envs = [gym.make(env_name) for _ in range(n_workers)] 

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

except KeyboardInterrupt: 
    print("Main thread received KeyboardInterrupt! Setting training_finished flag...")
    training_finished = True 

finally:
    print("Sending sentinel value to gradient queue...")
    gradient_queue.put(None) 
    gradient_applier_thread.join(timeout=5) 
    if gradient_applier_thread.is_alive(): 
        print("Warning: Gradient applier thread did not finish within timeout.")
    print("Closing environments...")
    for env in envs: 
        env.close()
    print("Program finished.")