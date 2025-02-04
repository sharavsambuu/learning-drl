import os
import random
import math
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


debug_render   = True
debug          = True
play_frequency = 50
num_episodes   = 1500
learning_rate  = 0.001
gamma          = 0.99
env_name       = "CartPole-v1"
group_size     = 2     
max_steps      = 500


class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=self.n_actions)(activation_layer_2)
        output_layer       = nn.softmax(output_dense_layer)
        return output_layer


env_array   = [gym.make(env_name, render_mode=None) for _ in range(group_size)]
env         = gym.make(env_name, render_mode='human')
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n


actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params
actor_optimizer_def    = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def .init(actor_model_params )


def calculate_grpo(states):
    return 0.4 

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, props): 
    # props[0] - states
    # props[1] - discounted_rewards
    # props[2] - actions
    baselines   = jnp.asarray([calculate_grpo(state) for state in props[0]]) 
    advantages  = props[1] - baselines # Use baseline instead of Critic values
    def loss_fn(params):
        action_probabilities = actor_module.apply({'params': params}, props[0])
        probabilities        = gather(action_probabilities, props[2])
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_actor_model_params = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_actor_model_params, loss


def rollout_trajectory(group_member_id, actor_model_params):

    group_reward   = 0.0

    states         = np.zeros(shape=(max_steps,) + state_shape, dtype=np.float32) 
    actions        = np.zeros(shape=(max_steps, 1), dtype=np.float32)
    rewards        = np.zeros(shape=(max_steps, 1), dtype=np.float32)
    dones          = np.zeros(shape=(max_steps, 1), dtype=np.float32)
    step           = 0

    env            = env_array[group_member_id]
    state, info    = env.reset()
    state          = np.array(state, dtype=np.float32)

    for _ in range(max_steps):
        action_probabilities = actor_inference(actor_model_params, jnp.asarray([state]))
        action_probabilities = np.array(action_probabilities[0])
        action               = np.random.choice(n_actions, p=action_probabilities)
        next_state, reward, terminated, truncated, info = env.step(int(action))
        done              = terminated or truncated
        next_state        = np.array(next_state, dtype=np.float32)
        states [step, :]  = state
        actions[step   ]  = action
        rewards[step   ]  = reward
        dones  [step   ]  = float(done)
        step             +=1
        if done:
            trajectory_length  = step
            discounted_rewards = np.zeros_like(rewards[:-trajectory_length])
            for t in range(0, trajectory_length):
                G_t = 0.0
                for idx, j in enumerate(range(t, trajectory_length)):
                    G_t += (gamma**idx)*rewards[j]*(1.0-dones[j])
                discounted_rewards[t] = G_t
            discounted_rewards = (discounted_rewards-np.mean(discounted_rewards))/(np.std(discounted_rewards) +1e-8)
            group_reward = discounted_rewards[0]

            print(f"GROUP-{group_member_id} REWARD IS {group_reward}, STEPS {trajectory_length}")
            break

    return group_member_id, group_reward, step, states, actions, rewards, dones

    pass

def rollout_group(actor_model_params):
    for group_member_id in range(group_size):
        rollout_trajectory(group_member_id=group_member_id, actor_model_params=actor_model_params)
    pass


try:
    for episode in range(num_episodes):

        rollout_group(actor_model_params=actor_model_params)

        if episode%play_frequency==0 and debug_render==True:
            state, info = env.reset()
            state = np.array(state, dtype=np.float32)
            print(state.shape)
            states, actions, rewards, dones = [], [], [], []
            while True:
                action_probabilities  = actor_inference(actor_model_params, jnp.asarray([state]))
                action_probabilities  = np.array(action_probabilities[0])
                action                = np.random.choice(n_actions, p=action_probabilities)
                next_state, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)
                states .append(state    )
                actions.append(action   )
                rewards.append(reward   )
                dones  .append(int(done))
                state = next_state

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
                    discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5)
                    actor_optimizer_state, actor_model_params, _  = backpropagate_actor(
                        actor_optimizer_state,
                        actor_model_params   ,
                        (
                            jnp.asarray(states),
                            jnp.asarray(discounted_rewards),
                            jnp.asarray(actions)
                        )
                    )
                    break
finally:
    env.close()
    for env_item in env_array:
        env_item.close()