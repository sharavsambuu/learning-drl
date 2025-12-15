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
        return output_dense_layer  # logits (no softmax here)

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=1)(activation_layer_2)
        return output_dense_layer


env   = gym.make('CartPole-v1', render_mode='human')
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

actor_optimizer_state  = actor_optimizer_def .init(actor_model_params)
critic_optimizer_state = critic_optimizer_def.init(critic_model_params)


@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)  # logits

@jax.jit
def critic_inference(params, x):
    return critic_module.apply({'params': params}, x)

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props):
    # props[0] - state
    # props[1] - next_state
    # props[2] - reward
    # props[3] - done
    next_value = jax.lax.stop_gradient(
        critic_module.apply({'params': critic_model_params}, jnp.asarray([props[1]]))[0][0]
    )

    def loss_fn(params):
        value     = critic_module.apply({'params': params}, jnp.asarray([props[0]]))[0][0]
        td_target = props[2] + (gamma*next_value)*(1-props[3])
        td_error  = jax.lax.stop_gradient(td_target) - value
        return jnp.square(td_error)

    loss, gradients              = jax.value_and_grad(loss_fn)(critic_model_params)
    updates, new_optimizer_state = critic_optimizer_def.update(gradients, optimizer_state, critic_model_params)
    new_critic_model_params      = optax.apply_updates(critic_model_params, updates)
    return new_optimizer_state, new_critic_model_params, loss

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, critic_model_params, props):
    # props[0] - state
    # props[1] - next_state
    # props[2] - reward
    # props[3] - done
    # props[4] - action
    value = jax.lax.stop_gradient(
        critic_module.apply({'params': critic_model_params}, jnp.asarray([props[0]]))[0][0]
    )
    next_value = jax.lax.stop_gradient(
        critic_module.apply({'params': critic_model_params}, jnp.asarray([props[1]]))[0][0]
    )
    advantage = props[2] + (gamma*next_value)*(1-props[3]) - value  # TD advantage

    def loss_fn(params, advantage_val):
        logits        = actor_module.apply({'params': params}, jnp.asarray([props[0]]))[0]
        log_probs     = jax.nn.log_softmax(logits, axis=-1)
        log_prob_act  = log_probs[props[4]]
        return -log_prob_act * advantage_val   # -(log pi(a|s)) * A

    loss, gradients              = jax.value_and_grad(loss_fn)(actor_model_params, advantage)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_actor_model_params       = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_actor_model_params, loss


global_step = 0

try:
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_rewards = []

        while True:
            global_step = global_step+1

            logits = actor_inference(actor_model_params, jnp.asarray([state]))
            probs  = jax.nn.softmax(logits, axis=-1)[0]
            probs  = np.array(probs)
            probs  = probs / probs.sum()  # safe normalize

            action = np.random.choice(n_actions, p=probs)

            next_state, reward, terminated, truncated, info = env.step(int(action))
            done       = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            episode_rewards.append(reward)

            actor_optimizer_state, actor_model_params, _  = backpropagate_actor(
                    actor_optimizer_state,
                    actor_model_params   ,
                    critic_model_params  ,
                    (state, next_state, reward, int(done), action)
                    )
            critic_optimizer_state, critic_model_params, _ = backpropagate_critic(
                    critic_optimizer_state,
                    critic_model_params   ,
                    (state, next_state, reward, int(done))
                    )

            state = next_state

            if debug_render:
                env.render()

            if done:
                print(episode, " - reward :", sum(episode_rewards))
                break
finally:
    env.close()
