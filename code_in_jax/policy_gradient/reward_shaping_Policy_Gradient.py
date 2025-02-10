#
# Reward shaping is techniques to give extra guidance in the sparse reward env.
# This adjustment preserves the optimal policy while giving the agent more frequent signals.
#
#   r'(s,a,s')=r(s,a,s')+γϕ(s')−ϕ(s)
#
#
# Potential Function :
# Is it one way to help on reward shaping 
#
#   potential(state, goal) 
# 
# returns the negative Manhattan distance or L1 norm
# so being closer to the goal yields a higher (less negative) value.
#
#


import random
import math
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax

debug_render                 = True
debug                        = True
num_episodes                 = 5000
learning_rate                = 0.001
gamma                        = 0.99  # Discount factor
sparse_reward_threshold      = 50    # Minimum steps to get a reward
sparse_reward_value          = 10    # Reward value for successful episodes

potential_scale_factor_start = 1.0
potential_scale_factor_end   = 0.1
decay_episodes_potential     = num_episodes // 3 

epsilon_start                = 1.0
epsilon_end                  = 0.01
decay_episodes_epsilon       = num_episodes // 2 


class SparseCartPoleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps"  : 50,
    }
    def __init__(self, render_mode='human', sparse_reward_threshold=40, sparse_reward_value=10):
        super().__init__()
        self.env = gym.make('CartPole-v1', render_mode=render_mode if debug_render else None)
        self.sparse_reward_threshold = sparse_reward_threshold
        self.sparse_reward_value     = sparse_reward_value
        self.current_steps           = 0
        self.action_space            = self.env.action_space
        self.observation_space       = self.env.observation_space
    def reset(self, seed=None, options=None):
        self.current_steps = 0
        observation, info  = self.env.reset(seed=seed, options=options)
        return np.array(observation, dtype=np.float32), info
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps += 1
        sparse_reward = 0
        done = terminated or truncated
        if terminated and self.current_steps >= self.sparse_reward_threshold:
            sparse_reward = self.sparse_reward_value
            if debug:
                print(f"Sparse reward given: {sparse_reward} at step {self.current_steps}")
        return np.array(observation, dtype=np.float32), sparse_reward, terminated, truncated, info
    def render(self, mode='human'):
        return self.env.render()
    def close(self):
        self.env.close()


class PolicyNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)


env         = SparseCartPoleEnv(render_mode='human', sparse_reward_threshold=sparse_reward_threshold, sparse_reward_value=sparse_reward_value)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)

n_actions             = env.action_space.n
pg_module             = PolicyNetwork(n_actions=n_actions)
dummy_input           = jnp.zeros(state.shape)
params                = pg_module.init(jax.random.PRNGKey(0), dummy_input)['params']
policy_network_params = params
optimizer_def         = optax.adam(learning_rate)
optimizer_state       = optimizer_def.init(policy_network_params)


def potential(state, scale_factor, angle_threshold=0.2):
    #
    # Scale factor behaviour
    #   0.0 - No shaping, baseline with just sparse reward
    #   0.1 - Very weak shaping
    #   0.5 - Weak shaping
    #   1.0 - Moderate shaping - Current baseline
    #   2.0 - Strong shaping
    #   5.0 - Very strong shaping
    #
    angle = state[2] # pole angle
    if abs(angle) < angle_threshold:
        return scale_factor * (jnp.cos(angle) - 1)
    else:
        return -0.001 

@jax.jit
def policy_inference(params, x):
    return pg_module.apply({'params': params}, x)

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def train_step(optimizer_state, policy_network_params, batch):
    def loss_fn(params):
        action_probabilities_list = pg_module.apply({'params': params}, batch[0])
        picked_action_probabilities = gather(action_probabilities_list, batch[1])
        log_probabilities = jnp.log(picked_action_probabilities)
        losses = jnp.multiply(log_probabilities, batch[2])
        return -jnp.sum(losses)
    loss, gradients = jax.value_and_grad(loss_fn)(policy_network_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, policy_network_params)
    new_policy_network_params = optax.apply_updates(policy_network_params, updates)
    return new_optimizer_state, new_policy_network_params, loss


global_steps            = 0
episode_rewards         = []
potential_scale_factor  = potential_scale_factor_start 
epsilon                 = epsilon_start 

try:
    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], []
        state, info    = env.reset()
        state          = np.array(state, dtype=np.float32)
        episode_reward = 0
        while True:
            global_steps += 1
            action_probabilities = policy_inference(policy_network_params, jnp.asarray([state]))
            action_probabilities = np.array(action_probabilities[0])
            action_probabilities /= action_probabilities.sum()

            if random.random() < epsilon:
                action = env.action_space.sample() # Exploration
            else:
                action = np.random.choice(n_actions, p=action_probabilities) 

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done      = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            # Apply potential-based reward shaping with tunable and decaying scale factor:
            shaped_reward   = reward + gamma * potential(new_state, potential_scale_factor) - potential(state, potential_scale_factor)
            episode_reward += shaped_reward

            states .append(state)
            actions.append(action)
            rewards.append(shaped_reward)
            dones  .append(done)

            state = new_state

            if debug_render:
                env.render()

            if done:
                episode_rewards.append(episode_reward)
                if debug:
                    if episode_reward > 0:
                        print(f"{episode} - Sparse Reward Success! Total shaped reward: {episode_reward}, Steps: {len(rewards)}")
                    else:
                        print(f"{episode} - Sparse Reward Failure. Total shaped reward: {episode_reward}, Steps: {len(rewards)}")

                episode_length     = len(rewards)
                discounted_rewards = np.zeros_like(rewards)
                for t in range(episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t += (gamma ** idx) * rewards[j] * (1 - dones[j])
                    discounted_rewards[t] = G_t
                discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-5)

                optimizer_state, policy_network_params, loss = train_step(
                    optimizer_state,
                    policy_network_params,
                    (
                        jnp.asarray(states),
                        jnp.asarray(actions),
                        jnp.asarray(discounted_rewards)
                    )
                )

                if debug:
                    print(f"Episode {episode}: Loss = {loss:.4f}")
                break

        # potential_scale_factor with linear decay 
        if episode < decay_episodes_potential:
            decay_rate_potential = (potential_scale_factor_start - potential_scale_factor_end) / decay_episodes_potential
            potential_scale_factor = potential_scale_factor_start - decay_rate_potential * episode
            potential_scale_factor = max(potential_scale_factor, potential_scale_factor_end)
        else:
            potential_scale_factor = potential_scale_factor_end

        # epsilon decay to add exploration
        if episode < decay_episodes_epsilon:
            decay_rate_epsilon = (epsilon_start - epsilon_end) / decay_episodes_epsilon
            epsilon = epsilon_start - decay_rate_epsilon * episode
            epsilon = max(epsilon, epsilon_end)
        else:
            epsilon = epsilon_end


        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}: Average shaped reward over last 10 episodes = {avg_reward:.2f}, Potential Scale Factor: {potential_scale_factor:.3f}, Epsilon: {epsilon:.3f}")

finally:
    env.close()