#
# DDPG aka Deep Deterministic Policy Gradient (Pendulum-v1)
#
# Implemented:
#   - Deterministic Actor + Q Critic
#   - Experience Replay Buffer (deque)
#   - Ornstein-Uhlenbeck action noise for exploration
#   - Soft target updates (tau)
#
# Improvements:
#   1) ActorNetwork no longer depends on 'env' inside the module (env wasn't defined yet)
#      -> pass action_dim explicitly
#   2) update_critic() no longer uses target_actor_model_params as a hidden global
#      -> pass target_actor_model_params into the JIT function (correct + stable)
#   3) Action shapes are enforced as (action_dim,) float32 for env.step()
#   4) Batch dtypes are enforced (reward float32, done float32)
#   5) Target values are stop_gradient'ed (prevents accidental gradient paths)
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
actor_noise_std = 0.1    # Standard deviation of actor exploration noise


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
    action_dim       : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(x)     # Output action dimension
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


env               = gym.make('Pendulum-v1', render_mode='human' if debug_render else None)
state, info       = env.reset()
state             = np.array(state, dtype=np.float32)

action_dim        = env.action_space.shape[0]
action_space_high = float(env.action_space.high[0])
action_space_low  = float(env.action_space.low[0])


actor_module              = ActorNetwork(action_space_high=action_space_high, action_dim=action_dim)
actor_params              = actor_module.init(jax.random.PRNGKey(0), jnp.asarray([state], dtype=jnp.float32))['params']
actor_model_params        = actor_params
target_actor_model_params = actor_params

critic_module              = CriticNetwork()
critic_params              = critic_module.init(
    jax.random.PRNGKey(0),
    jnp.asarray([state], dtype=jnp.float32),
    jnp.zeros((1, action_dim), dtype=jnp.float32)
)['params']
critic_model_params        = critic_params
target_critic_model_params = critic_params


actor_optimizer_def    = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params)

critic_optimizer_def   = optax.adam(learning_rate)
critic_optimizer_state = critic_optimizer_def.init(critic_model_params)

replay_buffer = deque(maxlen=buffer_size)


@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def critic_inference(params, x, action):
    return critic_module.apply({'params': params}, x, action)

@jax.jit
def update_critic(
    critic_optimizer_state    ,
    critic_model_params       ,
    target_actor_model_params ,
    target_critic_model_params,
    batch                     ,
    gamma
):
    states, actions, rewards, next_states, dones = batch

    def critic_loss_fn(critic_params):
        next_actions    = actor_module.apply({'params': target_actor_model_params}, next_states)
        target_q_values = critic_module.apply({'params': target_critic_model_params}, next_states, next_actions).reshape(-1)

        target_values   = rewards + gamma * (1.0 - dones) * target_q_values
        target_values   = jax.lax.stop_gradient(target_values)

        q_values        = critic_module.apply({'params': critic_params}, states, actions).reshape(-1)
        critic_loss     = jnp.mean((q_values - target_values) ** 2)
        return critic_loss

    loss, gradients              = jax.value_and_grad(critic_loss_fn)(critic_model_params)
    updates, new_optimizer_state = critic_optimizer_def.update(gradients, critic_optimizer_state, critic_model_params)
    new_critic_model_params      = optax.apply_updates(critic_model_params, updates)
    return new_optimizer_state, new_critic_model_params, loss

@jax.jit
def update_actor(actor_optimizer_state, actor_model_params, critic_model_params, batch):
    states = batch[0]

    def actor_loss_fn(actor_params):
        actions    = actor_module.apply({'params': actor_params}, states)
        q_values   = critic_module.apply({'params': critic_model_params}, states, actions).reshape(-1)
        actor_loss = -jnp.mean(q_values)  # maximize Q
        return actor_loss

    loss, gradients              = jax.value_and_grad(actor_loss_fn)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, actor_optimizer_state, actor_model_params)
    new_actor_model_params       = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_actor_model_params, loss

@jax.jit
def soft_update(target_model_params, model_params, tau):
    new_target_params = jax.tree_util.tree_map(
        lambda target_params, params: tau * params + (1.0 - tau) * target_params,
        target_model_params, model_params
    )
    return new_target_params


global_steps = 0
try:
    for episode in range(num_episodes):
        state, info = env.reset()
        state       = np.array(state, dtype=np.float32)

        episode_rewards = []

        actor_noise = OUProcess(
            theta=0.15,
            mu=np.zeros(action_dim, dtype=np.float32),
            sigma=actor_noise_std,
            dt=1e-2,
            x0=np.zeros(action_dim, dtype=np.float32)
        )

        while True:
            global_steps += 1

            # Actor deterministic action (shape: (action_dim,))
            action_det  = actor_inference(actor_model_params, jnp.asarray([state], dtype=jnp.float32))[0]
            action_det  = np.array(action_det, dtype=np.float32)

            # OU noise (shape: (action_dim,))
            action_noise = actor_noise.sample().astype(np.float32)

            # Final action (clip to env bounds)
            action = action_det + action_noise
            action = np.clip(action, action_space_low, action_space_high).astype(np.float32)

            new_state, reward, terminated, truncated, info = env.step(action)
            done      = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            replay_buffer.append((state, action, float(reward), new_state, float(done)))

            state = new_state
            episode_rewards.append(reward)

            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)

                states      = np.array([sample[0] for sample in minibatch], dtype=np.float32)
                actions     = np.array([sample[1] for sample in minibatch], dtype=np.float32)
                rewards     = np.array([sample[2] for sample in minibatch], dtype=np.float32)
                next_states = np.array([sample[3] for sample in minibatch], dtype=np.float32)
                dones       = np.array([sample[4] for sample in minibatch], dtype=np.float32)

                batch = (
                    jnp.asarray(states, dtype=jnp.float32),
                    jnp.asarray(actions, dtype=jnp.float32),
                    jnp.asarray(rewards, dtype=jnp.float32),
                    jnp.asarray(next_states, dtype=jnp.float32),
                    jnp.asarray(dones, dtype=jnp.float32),
                )

                critic_optimizer_state, critic_model_params, critic_loss = update_critic(
                    critic_optimizer_state,
                    critic_model_params,
                    target_actor_model_params,
                    target_critic_model_params,
                    batch,
                    gamma
                )

                actor_optimizer_state, actor_model_params, actor_loss = update_actor(
                    actor_optimizer_state,
                    actor_model_params,
                    critic_model_params,
                    batch
                )

                target_critic_model_params = soft_update(target_critic_model_params, critic_model_params, tau)
                target_actor_model_params  = soft_update(target_actor_model_params , actor_model_params , tau)

            if debug_render:
                env.render()

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break
finally:
    env.close()
