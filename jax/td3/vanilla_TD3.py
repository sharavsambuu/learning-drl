#
#  TD3 aka Twin Delayed Deep Deterministic Policy Gradient
#

import os
import random
import math
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np


debug_render            = False 
num_episodes            = 500
learning_rate           = 0.0003  # Slightly lower learning rate for TD3
gamma                   = 0.99
tau                     = 0.005   # Soft update rate
buffer_size             = 10000
batch_size              = 64
actor_noise_std         = 0.1     # Exploration noise for the actor
target_policy_noise_std = 0.2     # Noise added to target policy for smoothing
target_noise_clip       = 0.5     # Clip range for target policy noise
policy_delay            = 2       # Policy update frequency (delayed updates)


class OUProcess(object): # Ornstein-Uhlenbeck process for action noise (same as DDPG)
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


class ActorNetwork(flax.nn.Module): # Same deterministic actor as DDPG
    def apply(self, x, action_space_high):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, env.action_space.shape[0])
        output_layer       = flax.nn.tanh(output_dense_layer) * action_space_high
        return output_layer

class CriticNetwork(flax.nn.Module): # Twin Critic Networks
    def apply(self, x, action): # No change in Critic Network architecture
        combined_input   = jnp.concatenate([x, action], -1)
        dense_layer_1      = flax.nn.Dense(combined_input, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(activation_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, 1)
        return output_layer


env   = gym.make('Pendulum-v1') # Continuous action space environment
state = env.reset()
action_space_high = env.action_space.high[0]

actor_module     = ActorNetwork.partial(action_space_high=action_space_high)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params)
target_actor_model = flax.nn.Model(actor_module, actor_params) # Target Actor Network
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)

critic_module_1       = CriticNetwork.partial() # Twin Critic 1
critic_module_2       = CriticNetwork.partial() # Twin Critic 2
_, critic_params_1    = critic_module_1.init_by_shape(jax.random.PRNGKey(0), [state.shape], [env.action_space.shape])
_, critic_params_2    = critic_module_2.init_by_shape(jax.random.PRNGKey(0), [state.shape], [env.action_space.shape])
critic_model_1        = flax.nn.Model(critic_module_1, critic_params_1)
critic_model_2        = flax.nn.Model(critic_module_2, critic_params_2)
target_critic_model_1 = flax.nn.Model(critic_module_1, critic_params_1) # Target Critic 1
target_critic_model_2 = flax.nn.Model(critic_module_2, critic_params_2) # Target Critic 2
critic_optimizer_1    = flax.optim.Adam(learning_rate).create(critic_model_1)
critic_optimizer_2    = flax.optim.Adam(learning_rate).create(critic_model_2)


replay_buffer = deque(maxlen=buffer_size)

@jax.jit
def actor_inference(model, x): # Actor inference remains same
    return model(x)

@jax.jit
def critic_inference(model, x, action): # Critic inference remains same
    return model(x, action)

@jax.jit
def update_critic(critic_optimizer_1, critic_optimizer_2, actor_model, critic_model_1, critic_model_2, target_critic_model_1, target_critic_model_2, batch, gamma, target_policy_noise_std, target_noise_clip, key):
    states, actions, rewards, next_states, dones = batch
    key1, key2 = jax.random.split(key) # Split key for noise

    def critic_loss_fn(critic_model_1, critic_model_2): # Loss for both critics
        next_actions          = target_actor_model(next_states) # Target actor network to get next actions
        # Target Policy Smoothing: Add clipped noise to target actions
        noise                 = jnp.clip(jax.random.normal(key1, next_actions.shape) * target_policy_noise_std, -target_noise_clip, target_noise_clip)
        noisy_next_actions    = jnp.clip(next_actions + noise, -action_space_high, action_space_high) # Clip noisy actions

        target_q_values_1     = target_critic_model_1(next_states, noisy_next_actions).reshape(-1)    # Target critic 1 Q-values
        target_q_values_2     = target_critic_model_2(next_states, noisy_next_actions).reshape(-1)    # Target critic 2 Q-values
        target_q_values       = jnp.minimum(target_q_values_1, target_q_values_2)                     # Clipped Double-Q: Use minimum of two target critics
        target_values         = rewards + gamma * (1 - dones) * target_q_values                       # Bellman equation
        q_values_1            = critic_model_1(states, actions).reshape(-1)                           # Current critic 1 Q-values
        q_values_2            = critic_model_2(states, actions).reshape(-1)                           # Current critic 2 Q-values
        critic1_loss          = jnp.mean((q_values_1 - target_values)**2)                             # MSE loss for critic 1
        critic2_loss          = jnp.mean((q_values_2 - target_values)**2)                             # MSE loss for critic 2
        return critic1_loss, critic2_loss 

    (critic1_loss, critic2_loss), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=False, argnums=(0,1))(critic_optimizer_1.target, critic_optimizer_2.target) # Get gradients for both critics
    critic_optimizer_1                             = critic_optimizer_1.apply_gradient(grads=critic_grads[0]) # Apply gradients to critic 1 optimizer
    critic_optimizer_2                             = critic_optimizer_2.apply_gradient(grads=critic_grads[1]) # Apply gradients to critic 2 optimizer
    return critic_optimizer_1, critic_optimizer_2, critic1_loss, critic2_loss

@jax.jit
def update_actor(actor_optimizer, actor_model, critic_model_1, batch): # Actor update remains similar
    states = batch[0]
    def actor_loss_fn(actor_model):
        actions    = actor_model(states)
        actor_loss = -jnp.mean(critic_model_1(states, actions)) # Actor loss based on critic 1 (can use critic 2 or min of both)
        return actor_loss
    actor_grads, actor_loss = jax.value_and_grad(actor_loss_fn)(actor_optimizer.target)
    actor_optimizer        = actor_optimizer.apply_gradient(actor_grads)
    return actor_optimizer, actor_loss

@jax.jit
def soft_update(target_model_params, model_params, tau): # Soft update remains same
    new_target_params = jax.tree_util.tree_map(
        lambda target_params, params: tau * params + (1 - tau) * target_params,
        target_model_params, model_params
    )
    return new_target_params


global_steps = 0
actor_update_steps = 0 # Counter for delayed actor updates
rng = jax.random.PRNGKey(0) # Random key for noise
try:
    for episode in range(num_episodes):
        state           = env.reset()
        episode_rewards = []
        actor_noise     = OUProcess(theta=0.15, mu=0.0, sigma=actor_noise_std, dt=1e-2, x0=np.zeros(env.action_space.shape[0])) # Ornstein-Uhlenbeck noise
        while True:
            global_steps += 1
            actor_update_steps += 1 # Increment actor update counter
            rng, key = jax.random.split(rng) # Split random key

            action_det    = actor_inference(actor_optimizer.target, jnp.asarray([state]))[0] # Deterministic action
            action_noise  = actor_noise.sample() # Sample noise
            action        = np.clip(action_det + action_noise, -action_space_high, action_space_high) # Add exploration noise

            new_state, reward, done, _ = env.step(action)

            replay_buffer.append((state, action, reward, new_state, done))
            state = new_state
            episode_rewards.append(reward)

            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                batch = [jnp.asarray(np.array([sample[i] for sample in minibatch])) for i in range(5)]

                critic_optimizer_1, critic_optimizer_2, critic_loss_1, critic_loss_2 = update_critic(critic_optimizer_1, critic_optimizer_2, actor_model, critic_model_1, critic_model_2, target_critic_model_1, target_critic_model_2, batch, gamma, target_policy_noise_std, target_noise_clip, key) # Update critics

                if actor_update_steps % policy_delay == 0: # Delayed policy updates - update actor every policy_delay steps
                    actor_optimizer, actor_loss = update_actor(actor_optimizer, actor_model, critic_model_1, batch) # Update actor (using critic 1)

                    # Soft target network updates (update target networks after actor update)
                    target_critic_model_params_1 = soft_update(target_critic_model_1.params, critic_optimizer_1.target.params, tau)
                    target_critic_model_params_2 = soft_update(target_critic_model_2.params, critic_optimizer_2.target.params, tau)
                    target_actor_model_params    = soft_update(target_actor_model.params,  actor_optimizer.target.params,  tau)
                    target_critic_model_1        = target_critic_model_1.replace(params=target_critic_model_params_1)
                    target_critic_model_2        = target_critic_model_2.replace(params=target_critic_model_params_2)
                    target_actor_model           = target_actor_model.replace(params=target_actor_model_params)


            if debug_render:
                env.render()

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break
finally:
    env.close()

