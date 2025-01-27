#
# DDPG aka Deep Deterministic Policy Gradient 
#
#

import os
import random
import math
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np


debug_render  = False # Set to True to render environment (slows down training)
num_episodes  = 500
learning_rate = 0.0005
gamma         = 0.99
tau           = 0.005 # Soft update coefficient
buffer_size   = 10000
batch_size    = 64
actor_noise_std = 0.1 # Standard deviation of actor exploration noise


class OUProcess(object): # Ornstein-Uhlenbeck process for action noise
    def __init__(self, theta, mu, sigma, dt, x0=None):
        self.theta = theta
        self.mu    = mu
        self.sigma = sigma
        self.dt    = dt
        self.x0    = x0
        self.reset()
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    def sample(self):
        x        = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class ActorNetwork(flax.nn.Module): # Deterministic Actor Network
    def apply(self, x, action_space_high):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, env.action_space.shape[0]) # Output action dimension
        output_layer       = flax.nn.tanh(output_dense_layer) * action_space_high # Scale action to action space
        return output_layer

class CriticNetwork(flax.nn.Module): # Q-Value Critic Network
    def apply(self, x, action):
        combined_input   = jnp.concatenate([x, action], -1) # State and action as input
        dense_layer_1      = flax.nn.Dense(combined_input, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, 1) # Output single Q-value
        return output_layer

env   = gym.make('Pendulum-v1') # Continuous action space environment
state = env.reset()
action_space_high = env.action_space.high[0] # Get action space max value

actor_module     = ActorNetwork.partial(action_space_high=action_space_high)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params)
target_actor_model = flax.nn.Model(actor_module, actor_params) # Target Actor Network
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape], [env.action_space.shape]) # Critic takes state and action
critic_model     = flax.nn.Model(critic_module, critic_params)
target_critic_model = flax.nn.Model(critic_module, critic_params) # Target Critic Network
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model)

replay_buffer = deque(maxlen=buffer_size) # Experience Replay Buffer

@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def critic_inference(model, x, action):
    return model(x, action)

@jax.jit
def update_critic(critic_optimizer, actor_model, critic_model, target_critic_model, batch, gamma):
    states, actions, rewards, next_states, dones = batch

    def critic_loss_fn(critic_model):
        next_actions           = target_actor_model(next_states) # Target actor network to get next actions
        target_q_values        = target_critic_model(next_states, next_actions).reshape(-1) # Target critic network to get target Q-values
        target_values          = rewards + gamma * (1 - dones) * target_q_values # Bellman equation for target values
        q_values             = critic_model(states, actions).reshape(-1) # Current critic Q-values
        critic_loss          = jnp.mean((q_values - target_values)**2) # MSE loss
        return critic_loss
    critic_grads, critic_loss = jax.value_and_grad(critic_loss_fn)(critic_optimizer.target)
    critic_optimizer         = critic_optimizer.apply_gradient(critic_grads)
    return critic_optimizer, critic_loss

@jax.jit
def update_actor(actor_optimizer, actor_model, critic_model, batch):
    states = batch[0]
    def actor_loss_fn(actor_model):
        actions    = actor_model(states) # Actor generates actions
        actor_loss = -jnp.mean(critic_model(states, actions)) # Actor loss is to maximize Critic Q-value
        return actor_loss
    actor_grads, actor_loss = jax.value_and_grad(actor_loss_fn)(actor_optimizer.target)
    actor_optimizer        = actor_optimizer.apply_gradient(actor_grads)
    return actor_optimizer, actor_loss

@jax.jit
def soft_update(target_model_params, model_params, tau): # Soft target network update
    new_target_params = jax.tree_util.tree_map(
        lambda target_params, params: tau * params + (1 - tau) * target_params,
        target_model_params, model_params
    )
    return new_target_params

global_steps = 0
try:
    for episode in range(num_episodes):
        state           = env.reset()
        episode_rewards = []
        actor_noise     = OUProcess(theta=0.15, mu=0.0, sigma=actor_noise_std, dt=1e-2, x0=np.zeros(env.action_space.shape[0])) # Ornstein-Uhlenbeck noise process
        while True:
            global_steps += 1
            action_det    = actor_inference(actor_optimizer.target, jnp.asarray([state]))[0] # Deterministic action from actor
            action_noise  = actor_noise.sample() # Sample noise
            action        = np.clip(action_det + action_noise, -action_space_high, action_space_high) # Add noise for exploration

            new_state, reward, done, _ = env.step(action)

            replay_buffer.append((state, action, reward, new_state, done)) # Store experience in replay buffer
            state = new_state
            episode_rewards.append(reward)

            if len(replay_buffer) > batch_size: # Train when replay buffer is large enough
                minibatch = random.sample(replay_buffer, batch_size)
                batch = [jnp.asarray(np.array([sample[i] for sample in minibatch])) for i in range(5)] # Prepare batch

                critic_optimizer, critic_loss = update_critic(critic_optimizer, actor_model, critic_model, target_critic_model, batch, gamma) # Update critic
                actor_optimizer, actor_loss   = update_actor(actor_optimizer, actor_model, critic_model, batch) # Update actor

                # Soft target network updates
                target_critic_model_params = soft_update(target_critic_model.params, critic_optimizer.target.params, tau)
                target_actor_model_params  = soft_update(target_actor_model.params,  actor_optimizer.target.params,  tau)
                target_critic_model        = target_critic_model.replace(params=target_critic_model_params)
                target_actor_model         = target_actor_model.replace(params=target_actor_model_params)


            if debug_render:
                env.render()

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break
finally:
    env.close()
