import os
import random
import math
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


debug_render        = True
debug               = False
num_episodes        = 1500
learning_rate       = 0.001
gamma               = 0.99
entropy_coefficient = 0.01


class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        x = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(x)
        output_dense_layer = nn.Dense(features=self.n_actions)(activation_layer_2)
        output_layer       = nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        x = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(x)
        output_dense_layer = nn.Dense(features=1)(activation_layer_2)
        return output_dense_layer


env   = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()
state = np.array(state, dtype=np.float32)

n_actions        = env.action_space.n

actor_module     = ActorNetwork(n_actions=n_actions)
dummy_input      = jnp.zeros(state.shape)
actor_params     = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model      = actor_params

critic_module    = CriticNetwork()
critic_params    = critic_module.init(jax.random.PRNGKey(1), dummy_input)['params']
critic_model     = critic_params

actor_optimizer_def    = optax.adam(learning_rate)
critic_optimizer_def   = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def.init(actor_model)
critic_optimizer_state = critic_optimizer_def.init(critic_model)


@jax.jit
def actor_inference(actor_params, x):
    return actor_module.apply({'params': actor_params}, x)

@jax.jit
def critic_inference(critic_params, x):
    return critic_module.apply({'params': critic_params}, x).squeeze(-1)

@jax.jit
def backpropagate_critic(critic_optimizer_state, critic_model, actor_model, batch):
    states, discounted_rewards = batch
    def critic_loss_fn(critic_params):
        values      = critic_module.apply({'params': critic_params}, states).squeeze(-1)
        advantages  = discounted_rewards - values
        return jnp.mean(jnp.square(advantages))
    critic_loss, critic_gradients = jax.value_and_grad(critic_loss_fn)(critic_model)
    critic_updates, new_critic_optimizer_state = critic_optimizer_def.update(
        critic_gradients, critic_optimizer_state, critic_model
    )
    new_critic_model = optax.apply_updates(critic_model, critic_updates)
    return new_critic_optimizer_state, new_critic_model, critic_loss

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(actor_optimizer_state, actor_model, critic_model, batch):
    states, discounted_rewards, actions = batch
    values      = jax.lax.stop_gradient(critic_module.apply({'params': critic_model}, states).squeeze(-1))
    advantages  = jnp.subtract(discounted_rewards, values)

    def actor_loss_fn(actor_params):
        action_probabilities      = actor_module.apply({'params': actor_params}, states)
        probabilities             = gather(action_probabilities, actions)
        log_probabilities         = jnp.log(probabilities)

        entropies                 = -jnp.sum(
                    action_probabilities * jnp.log(action_probabilities + 1e-8),
                    axis = 1
                )
        actor_loss = -jnp.mean(log_probabilities * (advantages + entropy_coefficient * entropies)) # Corrected loss with negative sign
        return actor_loss

    actor_loss, actor_gradients = jax.value_and_grad(actor_loss_fn)(actor_model)
    actor_updates, new_actor_optimizer_state = actor_optimizer_def.update(
        actor_gradients, actor_optimizer_state, actor_model
    )
    new_actor_model = optax.apply_updates(actor_model, actor_updates)
    return new_actor_optimizer_state, new_actor_model, actor_loss


global_step = 0

try:
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        states, actions, rewards, dones = [], [], [], []
        while True:
            global_step = global_step+1

            action_probabilities  = actor_inference(actor_model, jnp.asarray([state]))
            action_probabilities  = np.array(action_probabilities[0])
            action                = np.random.choice(n_actions, p=action_probabilities)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(int(done))

            state = new_state

            if debug_render:
                env.render()

            if done:
                print(f"{episode} - reward : {sum(rewards)}")

                episode_length = len(rewards)

                discounted_rewards = np.zeros_like(rewards)
                for t in range(episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                    discounted_rewards[t] = G_t
                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5)

                actor_optimizer_state, actor_model, actor_loss  = backpropagate_actor(
                    actor_optimizer_state,
                    actor_model,
                    critic_model,
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                        jnp.asarray(actions)
                    )
                )

                critic_optimizer_state, critic_model, critic_loss = backpropagate_critic(
                    critic_optimizer_state,
                    critic_model,
                    actor_model,
                    (
                        jnp.asarray(states),
                        jnp.asarray(discounted_rewards),
                    )
                )

                break
finally:
    env.close()