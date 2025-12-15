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


class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=self.n_actions)(activation_layer_2)
        return output_dense_layer   # logits (no softmax here)

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=1)(activation_layer_2)
        return output_dense_layer


env         = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n


actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

critic_module          = CriticNetwork()
critic_params          = critic_module.init(jax.random.PRNGKey(0), dummy_input)['params']
critic_model_params    = critic_params


actor_optimizer_def    = optax.adam(learning_rate)
critic_optimizer_def   = optax.adam(learning_rate)

actor_optimizer_state  = actor_optimizer_def .init(actor_model_params )
critic_optimizer_state = critic_optimizer_def.init(critic_model_params)


@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)   # logits

@jax.jit
def critic_inference(params, x):
    return critic_module.apply({'params': params}, x)

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props):
    # props[0] - states
    # props[1] - discounted_rewards
    def loss_fn(params):
        values     = critic_module.apply({'params': params}, props[0])
        values     = jnp.reshape(values, (values.shape[0],))
        advantages = props[1] - values
        return jnp.mean(jnp.square(advantages))
    loss, gradients              = jax.value_and_grad(loss_fn)(critic_model_params)
    updates, new_optimizer_state = critic_optimizer_def.update(gradients, optimizer_state, critic_model_params)
    new_critic_model_params      = optax.apply_updates(critic_model_params, updates)
    return new_optimizer_state, new_critic_model_params, loss

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, critic_model_params, props):
    # props[0] - states
    # props[1] - discounted_rewards
    # props[2] - actions
    values     = jax.lax.stop_gradient(critic_module.apply({'params': critic_model_params}, props[0]))
    values     = jnp.reshape(values, (values.shape[0],))
    advantages = props[1] - values

    def loss_fn(params):
        logits     = actor_module.apply({'params': params}, props[0])
        log_probs  = jax.nn.log_softmax(logits, axis=-1)
        actions    = props[2].astype(jnp.int32)
        picked_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(1)
        # loss = -E[ log pi(a|s) * advantage ]
        return -jnp.mean(jnp.multiply(picked_log_probs, advantages))

    loss, gradients              = jax.value_and_grad(loss_fn)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_actor_model_params       = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_actor_model_params, loss


global_step = 0

try:
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        states, actions, rewards, dones = [], [], [], []

        while True:
            global_step = global_step+1

            logits = actor_inference(actor_model_params, jnp.asarray([state]))
            probs  = jax.nn.softmax(logits, axis=-1)[0]
            probs  = np.array(probs)
            probs  = probs / probs.sum()

            action = np.random.choice(n_actions, p=probs)

            next_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            states .append(state    )
            actions.append(action   )
            rewards.append(reward   )
            dones  .append(int(done))

            state = next_state

            if debug_render:
                env.render()

            if done:
                print(episode, " - reward :", sum(rewards))

                episode_length = len(rewards)

                # O(T) discounted returns (reverse)
                discounted_rewards = np.zeros(episode_length, dtype=np.float32)
                running_return = 0.0
                for t in reversed(range(episode_length)):
                    if dones[t]:
                        running_return = 0.0
                    running_return        = rewards[t] + gamma * running_return
                    discounted_rewards[t] = running_return

                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5)

                actor_optimizer_state, actor_model_params, _  = backpropagate_actor(
                    actor_optimizer_state,
                    actor_model_params   ,
                    critic_model_params  ,
                    (
                        jnp.asarray(states                               ),
                        jnp.asarray(discounted_rewards, dtype=jnp.float32),
                        jnp.asarray(actions           , dtype=jnp.int32  )
                    )
                )

                critic_optimizer_state, critic_model_params, _ = backpropagate_critic(
                    critic_optimizer_state,
                    critic_model_params,
                    (
                        jnp.asarray(states                               ),
                        jnp.asarray(discounted_rewards, dtype=jnp.float32),
                    )
                )

                break
finally:
    env.close()
