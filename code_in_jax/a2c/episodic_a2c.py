import os
import random
import math
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


debug_render  = True
debug         = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99


class ActorNetwork(nn.Module): # Use flax.linen.Module
    n_actions: int # Define n_actions as a field

    @nn.compact # Use @nn.compact
    def __call__(self, x): # Use __call__ instead of apply
        x = nn.Dense(features=64)(x) # Use features instead of out_dim
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1) # Use features instead of out_dim
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=self.n_actions)(activation_layer_2) # Use features instead of out_dim, and self.n_actions
        output_layer       = nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(nn.Module): # Use flax.linen.Module
    @nn.compact # Use @nn.compact
    def __call__(self, x): # Use __call__ instead of apply
        x = nn.Dense(features=64)(x) # Use features instead of out_dim
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1) # Use features instead of out_dim
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=1)(activation_layer_2) # Use features instead of out_dim
        return output_dense_layer


env   = gym.make('CartPole-v1', render_mode='human') # add render_mode for new gym
state, info = env.reset() # env.reset() now returns state and info
state = np.array(state, dtype=np.float32) # ensure state is float32

n_actions        = env.action_space.n


actor_module     = ActorNetwork(n_actions=n_actions) # Pass n_actions during module creation
dummy_input      = jnp.zeros(state.shape)
actor_params     = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params'] # Initialize with dummy input and access 'params'
actor_model_params = actor_params # for clarity, keep params in this variable

critic_module    = CriticNetwork()
critic_params    = critic_module.init(jax.random.PRNGKey(0), dummy_input)['params'] # Initialize with dummy input and access 'params'
critic_model_params = critic_params # for clarity, keep params in this variable


actor_optimizer_def  = optax.adam(learning_rate) # Use optax.adam
critic_optimizer_def = optax.adam(learning_rate) # Use optax.adam

actor_optimizer_state  = actor_optimizer_def.init(actor_model_params) # init optimizer state
critic_optimizer_state = critic_optimizer_def.init(critic_model_params) # init optimizer state


@jax.jit
def actor_inference(params, x): # pass params as argument
    return actor_module.apply({'params': params}, x) # use module.apply and pass params

@jax.jit
def critic_inference(params, x): # pass params as argument
    return critic_module.apply({'params': params}, x) # use module.apply and pass params

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props): # pass optimizer_state and model_params
    # props[0] - states
    # props[1] - discounted_rewards
    def loss_fn(params):
        values      = critic_module.apply({'params': params}, props[0]) # use module.apply and pass params
        values      = jnp.reshape(values, (values.shape[0],))
        advantages  = props[1] - values
        return jnp.mean(jnp.square(advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(critic_model_params)
    updates, new_optimizer_state = critic_optimizer_def.update(gradients, optimizer_state, critic_model_params) # use optimizer_def.update
    new_critic_model_params = optax.apply_updates(critic_model_params, updates) # use optax.apply_updates
    return new_optimizer_state, new_critic_model_params, loss # return updated params

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, critic_model_params, props): # pass optimizer_state, actor_model_params, critic_model_params
    # props[0] - states
    # props[1] - discounted_rewards
    # props[2] - actions
    values      = jax.lax.stop_gradient(critic_module.apply({'params': critic_model_params}, props[0])) # use module.apply and pass critic_model_params
    values      = jnp.reshape(values, (values.shape[0],))
    advantages  = props[1] - values
    def loss_fn(params):
        action_probabilities = actor_module.apply({'params': params}, props[0]) # use module.apply and pass params
        probabilities        = gather(action_probabilities, props[2])
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params) # use optimizer_def.update
    new_actor_model_params = optax.apply_updates(actor_model_params, updates) # use optax.apply_updates
    return new_optimizer_state, new_actor_model_params, loss # return updated params


global_step = 0

try:
    for episode in range(num_episodes):
        state, info = env.reset() # env.reset() now returns state and info
        state = np.array(state, dtype=np.float32) # ensure state is float32
        states, actions, rewards, dones = [], [], [], []
        while True:
            global_step = global_step+1

            action_probabilities  = actor_inference(actor_model_params, jnp.asarray([state])) # pass actor_model_params
            action_probabilities  = np.array(action_probabilities[0])
            action                = np.random.choice(n_actions, p=action_probabilities)

            next_state, reward, terminated, truncated, info = env.step(int(action)) # env.step returns new values
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32) # ensure next_state is float32

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(int(done))

            state = next_state

            if debug_render:
                env.render()

            if done:
                print(episode, " - reward :", sum(rewards))

                episode_length = len(rewards)

                discounted_rewards = np.zeros_like(rewards)
                for t in range(0, episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                    discounted_rewards[t] = G_t
                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5) # https://twitter.com/araffin2/tweet/1329382226421837825

                actor_optimizer_state, actor_model_params, _  = backpropagate_actor( # update optimizer_state and model_params
                    actor_optimizer_state,
                    actor_model_params,
                    critic_model_params,
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                        jnp.asarray(actions)
                    )
                )

                critic_optimizer_state, critic_model_params, _ = backpropagate_critic( # update optimizer_state and model_params
                    critic_optimizer_state,
                    critic_model_params,
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                    )
                )

                break
finally:
    env.close()