import os
import random
import math
import gymnasium  as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax

debug_render  = True
debug         = False
num_episodes  = 5000
learning_rate = 0.001
gamma         = 0.99   # discount factor

class PolicyNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        logits             = nn.Dense(features=self.n_actions)(activation_layer_2)  # logits (no softmax)
        return logits

env   = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state = np.array(state, dtype=np.float32)

n_actions             = env.action_space.n
pg_module             = PolicyNetwork(n_actions=n_actions)
dummy_input           = jnp.zeros(state.shape)
params                = pg_module.init(jax.random.PRNGKey(0), dummy_input)['params']
policy_network_params = params
optimizer_def         = optax.adam(learning_rate)
optimizer_state       = optimizer_def.init(policy_network_params)


@jax.jit
def policy_inference(params, x):
    return pg_module.apply({'params': params}, x)  # returns logits

@jax.vmap
def gather(log_probability_vec, action_index):
    return log_probability_vec[action_index]

@jax.jit
def train_step(optimizer_state, policy_network_params, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - discounted rewards 
    def loss_fn(params):
        logits_list             = pg_module.apply({'params': params}, batch[0])
        log_probs_list          = jax.nn.log_softmax(logits_list, axis=-1)
        picked_log_probabilities= gather(log_probs_list, batch[1])
        losses                  = jnp.multiply(picked_log_probabilities, batch[2])
        return -jnp.sum(losses)
    loss, gradients = jax.value_and_grad(loss_fn)(policy_network_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, policy_network_params)
    new_policy_network_params = optax.apply_updates(policy_network_params, updates)
    return new_optimizer_state, new_policy_network_params, loss


global_steps = 0
try:
    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], []
        state, info = env.reset()
        state       = np.array(state, dtype=np.float32)

        while True:
            global_steps = global_steps+1

            logits = policy_inference(policy_network_params, jnp.asarray([state]))
            probs  = jax.nn.softmax(logits, axis=-1)[0]
            probs  = np.array(probs)
            probs  = probs / probs.sum()
            action = np.random.choice(n_actions, p=probs)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            states .append(state )
            actions.append(action)
            rewards.append(reward)
            dones  .append(done  )

            state = new_state

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(rewards)))

                # O(T) discounted returns (reverse)
                episode_length     = len(rewards)
                discounted_rewards = np.zeros(episode_length, dtype=np.float32)

                running_return = 0.0
                for t in reversed(range(episode_length)):
                    if dones[t]:
                        running_return = 0.0
                    running_return = rewards[t] + gamma * running_return
                    discounted_rewards[t] = running_return

                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5)

                optimizer_state, policy_network_params, loss = train_step(
                    optimizer_state,
                    policy_network_params,
                    (
                        jnp.asarray(states                               ),
                        jnp.asarray(actions           , dtype=jnp.int32  ),
                        jnp.asarray(discounted_rewards, dtype=jnp.float32)
                    ))

                break
finally:
    env.close()
