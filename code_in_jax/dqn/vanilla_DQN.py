import os
import random
import math
import gymnasium   as gym # Changed to gymnasium
from collections import deque

import flax.linen as nn  # Use flax.linen for neural network definitions
import jax
from jax import numpy as jnp
import numpy as np
import optax # Use optax for optimizers
import time

debug_render  = True
debug         = False
num_episodes  = 200
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000
replay_memory = deque(maxlen=memory_length)

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99 # discount factor
hidden_size   = 64 # Define hidden_size


class DeepQNetwork(nn.Module): # Use flax.linen.Module
    n_actions: int # Define n_actions as a field

    @nn.compact # Use @nn.compact
    def __call__(self, x): # Use __call__ instead of apply
        x = nn.Dense(features=hidden_size)(x) # Use features instead of out_dim # Removed Embed layer as input is state, not token index
        x = nn.relu(x)
        x = nn.Dense(features=hidden_size//2)(x) # Added more layers to match original layers
        x = nn.relu(x)
        logits = nn.Dense(features=self.n_actions)(x)  # Output logits for vocabulary # Use features instead of out_dim, and self.n_actions
        return logits

class CriticNetwork(nn.Module): # Added CriticNetwork definition to avoid potential confusion, although it's not used in DQN directly
    @nn.compact
    def __call__(self, x): # Input is the state
        x = nn.Embed(num_embeddings=vocab_size, features=hidden_size, embedding_init=jax.nn.initializers.uniform())(x) # Embed input
        x = nn.Dense(features=hidden_size)(x) # Use features instead of out_dim
        x = nn.relu(x)
        value = nn.Dense(features=1)(x) # Output a single value
        return value


env   = gym.make('CartPole-v1', render_mode='human') # or 'rgb_array' if you want to handle rendering manually
state, info = env.reset() # env.reset() returns tuple now # info is returned now as well

n_actions        = env.action_space.n

dqn_module       = DeepQNetwork(n_actions=n_actions) # Instantiate module with n_actions
dummy_input      = jnp.zeros(state.shape) # Create dummy input with state shape
params           = dqn_module.init(jax.random.PRNGKey(0), dummy_input) # Initialize parameters
q_network_params = params['params'] # Access params dict
target_q_network_params = params['params'] # Initialize target network params

optimizer        = optax.adam(learning_rate) # Use optax.adam
opt_state        = optimizer.init(q_network_params) # Initialize optimizer state


@jax.jit
def policy(params, x): # Pass params explicitly
    predicted_q_values = dqn_module.apply({'params': params}, x) # Pass params as dict to apply
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


def q_learning_loss(q_value_vec, target_q_value_vec, action, reward, done): # No jax.jit here, let train_step jit it
    one_hot_actions = jax.nn.one_hot(action, n_actions) # Create one-hot action vector
    q_value         = jnp.sum(one_hot_actions*q_value_vec) # Index q_value using one-hot actions
    td_target       = reward + gamma*jnp.max(target_q_value_vec)*(1.-done)
    td_error        = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

q_learning_loss_vmap = jax.vmap(q_learning_loss, in_axes=(0, 0, 0, 0, 0), out_axes=0) # Apply vmap to q_learning_loss - Define vmapped version here


@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch): # Pass params and opt_state
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(params):
        predicted_q_values = dqn_module.apply({'params': params}, batch[0]) # Pass params as dict to apply
        target_q_values    = dqn_module.apply({'params': target_q_network_params}, batch[3]) # Pass target params as dict
        return jnp.mean(
                q_learning_loss_vmap( # Use the vmapped loss function here
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params) # Get updates and new opt_state
    q_network_params = optax.apply_updates(q_network_params, updates) # Apply updates to params
    return q_network_params, opt_state, loss # Return updated params, opt_state and loss



global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset() # env.reset() returns tuple now # info is returned now as well

        state = np.array(state, dtype=np.float32) # Ensure state is numpy array and float32
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(q_network_params, state) # Pass params to policy
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
                if debug:
                    #print("epsilon :", epsilon)
                    pass

            new_state, reward, terminated, truncated, info = env.step(int(action)) # env.step returns 5 values now
            done = terminated or truncated

            new_state = np.array(new_state, dtype=np.float32) # Ensure new_state is numpy array and float32

            replay_memory.append((state, action, reward, new_state, float(done))) # Store done as float

            # Хангалттай batch цугласан бол DQN сүлжээг сургах
            if (len(replay_memory)>batch_size):
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                q_network_params, opt_state, loss = train_step( # Update train_step return values
                                            q_network_params,
                                            target_q_network_params,
                                            opt_state,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур
                                                # төхөөрөмийн санах ойруу хуулах
                                                jnp.asarray(list(states)), # Convert tuple to list before asarray
                                                jnp.asarray(list(actions), dtype=jnp.int32), # Convert tuple to list before asarray, and ensure dtype is int32
                                                jnp.asarray(list(rewards), dtype=jnp.float32), # Convert tuple to list before asarray, and ensure dtype is float32
                                                jnp.asarray(list(next_states)), # Convert tuple to list before asarray
                                                jnp.asarray(list(dones), dtype=jnp.float32) # Convert tuple to list before asarray, and ensure dtype is float32
                                            )
                                        )

            episode_rewards.append(reward)
            state = new_state

            # Тодорхой алхам тутамд target неорон сүлжээний жингүүдийг сайжирсан хувилбараар солих
            if global_steps%sync_steps==0:
                target_q_network_params = q_network_params # Directly copy params
                if debug:
                    print("сайжруулсан жингүүдийг target неорон сүлжээрүү хууллаа")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()