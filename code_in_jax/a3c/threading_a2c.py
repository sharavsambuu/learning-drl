import os
import random
import math
import time
import threading
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


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


env       = gym.make(env_name)
state, info     = env.reset()
n_actions = env.action_space.n
env.close()

# Initialize Actor and Critic networks and optimizers
actor_module = ActorNetwork(n_actions=n_actions)
critic_module = CriticNetwork()

dummy_state = jnp.zeros(state.shape)

actor_params = actor_module.init(jax.random.PRNGKey(0), dummy_state)['params']
critic_params = critic_module.init(jax.random.PRNGKey(1), dummy_state)['params']

actor_optimizer = optax.adam(learning_rate).init(actor_params)
critic_optimizer = optax.adam(learning_rate).init(critic_params)


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
    updates, critic_optimizer = optax.adam(learning_rate).update(gradients, critic_optimizer, critic_params) # Use optimizer_def.update
    critic_params = optax.apply_updates(critic_params, updates) # Apply updates to parameters
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
    updates, actor_optimizer = optax.adam(learning_rate).update(gradients, actor_optimizer, actor_params) # Use optimizer_def.update
    actor_params = optax.apply_updates(actor_params, updates) # Apply updates to parameters
    return actor_params, actor_optimizer, loss



episode_count = 0
global_step   = 0

lock = threading.Lock()
training_finished = False # Global flag to signal workers to stop

def training_worker(env, thread_index):
    global actor_params # Access global parameters
    global actor_optimizer
    global critic_params # Access global parameters
    global critic_optimizer

    global episode_count
    global global_step
    global training_finished

    try: # Wrap worker loop in try-except for KeyboardInterrupt
        for episode in range(num_episodes):
            if training_finished: # Check the flag at the beginning of each episode
                break

            state, info = env.reset()
            states, actions, rewards, dones = [], [], [], []

            while True:
                global_step = global_step + 1

                action_probabilities  = actor_inference(actor_params, jnp.asarray(state)) # Pass state directly
                action_probabilities  = np.array(action_probabilities) # remove index [0]
                action                = np.random.choice(n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(int(done))

                state = next_state

                if debug_render and thread_index==0:
                    env.render()

                if done or training_finished: # Check the flag at each step
                    if done:
                        print("{} step, {} worker, {} episode, reward : {}".format(
                            global_step, thread_index, episode, sum(rewards)))

                        episode_length = len(rewards)

                        discounted_rewards = np.zeros_like(rewards, dtype=np.float32) # Ensure float32 for JAX
                        for t in range(episode_length): # Correct range
                            G_t = 0
                            for idx, j in enumerate(range(t, episode_length)):
                                G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                            discounted_rewards[t] = G_t

                        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards)+1e-8) # Added small epsilon for numerical stability

                        with lock:
                            if not training_finished: # Check again before updating to prevent race condition after Ctrl+C
                                actor_params, actor_optimizer, _  = backpropagate_actor( # Update global parameters
                                    actor_params,
                                    actor_optimizer,
                                    critic_params,
                                    jnp.asarray(np.array(states)), # Convert lists to numpy arrays then jax arrays
                                    jnp.asarray(np.array(discounted_rewards)),
                                    jnp.asarray(np.array(actions), dtype=jnp.int32) # Ensure actions are int32
                                )
                                critic_params, critic_optimizer, _ = backpropagate_critic( # Update global parameters
                                    critic_params,
                                    critic_optimizer,
                                    jnp.asarray(np.array(states)), # Convert lists to numpy arrays then jax arrays
                                    jnp.asarray(np.array(discounted_rewards)),
                                )
                                episode_count = episode_count + 1
                    break # Break out of the while True loop if done or training finished
        print("{} id-тэй thread ажиллаж дууслаа.".format(thread_index))
    except KeyboardInterrupt:
        print(f"Worker {thread_index} received KeyboardInterrupt. Exiting...")
        pass # Exit gracefully

envs = [gym.make(env_name) for _ in range(n_workers)] # Corrected range

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

except KeyboardInterrupt: # Catch KeyboardInterrupt in the main thread
    print("Main thread received KeyboardInterrupt. Shutting down training...")
    training_finished = True # Set the global flag to signal workers to stop

finally:
    print("Closing environments...")
    for env in envs:
        env.close()
    print("Environments closed.")
    print("Program finished.")