#
# DDPG aka Deep Deterministic Policy Gradient
#
#

import os
import random
import math
from collections import deque
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax

debug_render    = True
num_episodes    = 500
learning_rate   = 0.0005
gamma           = 0.99
tau             = 0.005  # Soft update coefficient
buffer_size     = 10000
batch_size      = 64
actor_noise_std = 0.1  # Standard deviation of actor exploration noise

class OUProcess(object):  # Ornstein-Uhlenbeck process for action noise
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
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class ActorNetwork(nn.Module):  # Deterministic Actor Network
    action_space_high: float
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=env.action_space.shape[0])(x)  # Output action dimension
        output = nn.tanh(x) * self.action_space_high  # Scale action to action space
        return output

class CriticNetwork(nn.Module):  # Q-Value Critic Network
    @nn.compact
    def __call__(self, x, action):
        x = jnp.concatenate([x, action], -1)  # State and action as input
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        output = nn.Dense(features=1)(x)  # Output single Q-value
        return output


env               = gym.make('Pendulum-v1', render_mode='human' if debug_render else None)  # Continuous action space environment
state, info       = env.reset()
state             = np.array(state, dtype=np.float32) # ensure state is float32
action_space_high = env.action_space.high[0]  # Get action space max value

actor_module               = ActorNetwork(action_space_high=action_space_high)
actor_params               = actor_module.init(jax.random.PRNGKey(0), jnp.asarray([state]))['params']
actor_model_params         = actor_params
target_actor_model_params  = actor_params

critic_module              = CriticNetwork()
critic_params              = critic_module.init(jax.random.PRNGKey(0), jnp.asarray([state]), jnp.zeros((1, *env.action_space.shape)))['params']  # Critic takes state and action
critic_model_params        = critic_params
target_critic_model_params = critic_params

actor_optimizer_def    = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params)

critic_optimizer_def   = optax.adam(learning_rate)
critic_optimizer_state = critic_optimizer_def.init(critic_model_params)

replay_buffer = deque(maxlen=buffer_size)  # Experience Replay Buffer


@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def critic_inference(params, x, action):
    return critic_module.apply({'params': params}, x, action)

@jax.jit
def update_critic(critic_optimizer_state, actor_model_params, critic_model_params, target_critic_model_params, batch, gamma):
    states, actions, rewards, next_states, dones = batch

    def critic_loss_fn(critic_params):
        next_actions    = actor_module.apply({'params': target_actor_model_params}, next_states)  # Target actor network to get next actions
        target_q_values = critic_module.apply({'params': target_critic_model_params}, next_states, next_actions).reshape(-1)  # Target critic network to get target Q-values
        target_values   = rewards + gamma * (1 - dones) * target_q_values  # Bellman equation for target values
        q_values        = critic_module.apply({'params': critic_params}, states, actions).reshape(-1)  # Current critic Q-values
        critic_loss     = jnp.mean((q_values - target_values)**2)  # MSE loss
        return critic_loss

    loss, gradients              = jax.value_and_grad(critic_loss_fn)(critic_model_params)
    updates, new_optimizer_state = critic_optimizer_def.update(gradients, critic_optimizer_state, critic_model_params)
    new_critic_model_params      = optax.apply_updates(critic_model_params, updates)
    return new_optimizer_state, new_critic_model_params, loss

@jax.jit
def update_actor(actor_optimizer_state, actor_model_params, critic_model_params, batch):
    states = batch[0]

    def actor_loss_fn(actor_params):
        actions = actor_module.apply({'params': actor_params}, states)  # Actor generates actions
        actor_loss = -jnp.mean(critic_module.apply({'params': critic_model_params}, states, actions))  # Actor loss is to maximize Critic Q-value
        return actor_loss

    loss, gradients = jax.value_and_grad(actor_loss_fn)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, actor_optimizer_state, actor_model_params)
    new_actor_model_params = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_actor_model_params, loss

@jax.jit
def soft_update(target_model_params, model_params, tau):  # Soft target network update
    new_target_params = jax.tree_util.tree_map(
        lambda target_params, params: tau * params + (1 - tau) * target_params,
        target_model_params, model_params
    )
    return new_target_params

global_steps = 0
try:
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32) # ensure state is float32
        episode_rewards = []
        actor_noise = OUProcess(theta=0.15, mu=np.zeros(env.action_space.shape[0]), sigma=actor_noise_std, dt=1e-2, x0=np.zeros(env.action_space.shape[0]))  # Ornstein-Uhlenbeck noise process
        while True:
            global_steps += 1
            action_det = actor_inference(actor_model_params, jnp.asarray([state]))[0]  # Deterministic action from actor
            action_noise = actor_noise.sample()  # Sample noise
            action = np.clip(action_det + action_noise, -action_space_high, action_space_high)  # Add noise for exploration

            new_state, reward, terminated, truncated, info = env.step(action)  # env.step returns new values
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32) # ensure new_state is float32

            replay_buffer.append((state, action, reward, new_state, done))  # Store experience in replay buffer
            state = new_state
            episode_rewards.append(reward)

            if len(replay_buffer) > batch_size:  # Train when replay buffer is large enough
                minibatch = random.sample(replay_buffer, batch_size)
                batch = [jnp.asarray(np.array([sample[i] for sample in minibatch])) for i in range(5)]  # Prepare batch

                critic_optimizer_state, critic_model_params, critic_loss = update_critic(
                    critic_optimizer_state, actor_model_params, critic_model_params, target_critic_model_params, batch, gamma
                )  # Update critic
                actor_optimizer_state, actor_model_params, actor_loss = update_actor(
                    actor_optimizer_state, actor_model_params, critic_model_params, batch
                )  # Update actor

                # Soft target network updates
                target_critic_model_params = soft_update(target_critic_model_params, critic_model_params, tau)
                target_actor_model_params = soft_update(target_actor_model_params, actor_model_params, tau)

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break
finally:
    env.close()