#
# PPO - It is basically Actor-Critic with carefully controlled policy updates to ensure stable and 
# reliable learning, preventing destabilization and wild oscillations during training.
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

debug_render    = False 
num_episodes    = 500
learning_rate   = 0.0003
gamma           = 0.99
gae_lambda      = 0.95
clip_ratio      = 0.2
policy_epochs   = 10
batch_size      = 64
mini_batch_size = 32      # For more stable training, split batch into mini-batches
sync_steps      = 100

class ActorNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(flax.nn.Module):
    def apply(self, x):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
        return output_dense_layer


env   = gym.make('CartPole-v1')
state = env.reset()
n_actions        = env.action_space.n

actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params)
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model     = flax.nn.Model(critic_module, critic_params)
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model)


@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def critic_inference(model, x):
    return model(x)

@jax.jit
def train_step(actor_optimizer, critic_optimizer, actor_model, critic_model, batch):
    states, actions, old_log_probs, advantages, returns = batch

    def actor_loss_fn(actor_model):
        action_probabilities = actor_model(states)
        log_probs = jnp.log(action_probabilities[jnp.arange(len(actions)), actions])
        ratio = jnp.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        return actor_loss

    def critic_loss_fn(critic_model):
        values = critic_model(states).reshape(-1)
        critic_loss = jnp.mean((values - returns)**2)
        return critic_loss

    actor_grads, actor_loss = jax.value_and_grad(actor_loss_fn)(actor_optimizer.target)
    critic_grads, critic_loss = jax.value_and_grad(critic_loss_fn)(critic_optimizer.target)

    actor_optimizer = actor_optimizer.apply_gradient(actor_grads)
    critic_optimizer = critic_optimizer.apply_gradient(critic_grads)

    return actor_optimizer, critic_optimizer, actor_loss, critic_loss


global_steps = 0
try:
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        episode_states, episode_actions, episode_rewards_list, episode_log_probs, episode_values = [], [], [], [], []
        while True:
            global_steps += 1
            action_probabilities = actor_inference(actor_optimizer.target, jnp.asarray([state]))
            action_probabilities = np.array(action_probabilities[0])
            action   = np.random.choice(n_actions, p=action_probabilities)
            value    = critic_inference(critic_optimizer.target, jnp.asarray([state]))

            log_prob = jnp.log(action_probabilities[action])

            new_state, reward, done, _ = env.step(int(action))

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_log_probs.append(log_prob)
            episode_values.append(value[0])

            state = new_state
            episode_rewards.append(reward)

            if debug_render:
                env.render()

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break

        # Calculate GAE advantages and returns
        values_np = np.array(episode_values + [0]) # Add bootstrap value
        advantages = np.zeros_like(episode_rewards_list, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(episode_rewards_list))):
            delta = episode_rewards_list[t] + gamma * values_np[t + 1] - values_np[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam

        returns = advantages + np.array(episode_values)

        # Prepare batch data
        batch_data = (
            np.array(episode_states),
            np.array(episode_actions),
            np.array(episode_log_probs),
            advantages,
            returns
        )

        # Train PPO for multiple epochs with mini-batches
        for _ in range(policy_epochs):
            perm = np.random.permutation(len(episode_states))
            for start_idx in range(0, len(episode_states), mini_batch_size):
                mini_batch_idx = perm[start_idx:start_idx + mini_batch_size]
                mini_batch = tuple(arr[mini_batch_idx] for arr in batch_data)
                actor_optimizer, critic_optimizer, actor_loss, critic_loss = train_step(
                    actor_optimizer, critic_optimizer, actor_model, critic_model, mini_batch
                )

finally:
    env.close()