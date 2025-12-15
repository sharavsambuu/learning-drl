#
# Reward shaping is techniques to give extra guidance in the sparse reward env.
# This adjustment preserves the optimal policy while giving the agent more frequent signals.
#
#   r'(s,a,s') = r(s,a,s') + γ * ϕ(s') − ϕ(s)
#
#
# - Potential function ϕ(s) should be time-invariant (not changing across training).
# - If you want shaping strength to decay, decay a multiplier OUTSIDE the potential:
#       shaped_reward = r + scale * ( γ*ϕ(s') − ϕ(s) )
#   so the shaping term fades out smoothly without redefining ϕ(s) itself.
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
gamma                        = 0.99                # Discount factor
sparse_reward_threshold      = 50                  # Minimum steps to get a reward
sparse_reward_value          = 10                  # Reward value for successful episodes

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
        logits = nn.Dense(features=self.n_actions)(x)  # logits (no softmax)
        return logits


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


def potential(state, angle_threshold=0.2):
    # Potential Function ϕ(s):
    # state -> potential (time-invariant).
    # Example: encourage pole angle closer to 0.
    angle = state[2] # pole angle
    if abs(angle) < angle_threshold:
        return (jnp.cos(angle) - 1.0)   # <= 0, best at angle=0
    else:
        return -0.001


@jax.jit
def policy_inference(params, x):
    return pg_module.apply({'params': params}, x)  # logits

@jax.vmap
def gather(log_probability_vec, action_index):
    return log_probability_vec[action_index]

@jax.jit
def train_step(optimizer_state, policy_network_params, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - discounted rewards
    def loss_fn(params):
        logits_list              = pg_module.apply({'params': params}, batch[0])
        log_probs_list           = jax.nn.log_softmax(logits_list, axis=-1)
        picked_log_probabilities = gather(log_probs_list, batch[1])
        losses                   = jnp.multiply(picked_log_probabilities, batch[2])
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

            logits = policy_inference(policy_network_params, jnp.asarray([state]))
            probs  = jax.nn.softmax(logits, axis=-1)[0]
            probs  = np.array(probs)
            probs  = probs / probs.sum()

            if random.random() < epsilon:
                action = env.action_space.sample() # Exploration
            else:
                action = np.random.choice(n_actions, p=probs)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done      = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            # ------------------------------------------------------------
            # Potential-Based Reward Shaping (PBRS):
            # shaped_reward = r + scale * ( γ*ϕ(s') − ϕ(s) )
            # ϕ is time-invariant. scale can decay safely.
            # ------------------------------------------------------------
            phi_s   = float(potential(state))
            phi_sp  = float(potential(new_state))
            shaping = potential_scale_factor * (gamma * phi_sp - phi_s)

            shaped_reward   = reward + shaping
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
                    if reward > 0:
                        print(f"{episode} - Sparse Reward Success! Total shaped reward: {episode_reward}, Steps: {len(rewards)}")
                    else:
                        print(f"{episode} - Sparse Reward Failure. Total shaped reward: {episode_reward}, Steps: {len(rewards)}")

                # O(T) discounted returns (reverse)
                episode_length     = len(rewards)
                discounted_rewards = np.zeros(episode_length, dtype=np.float32)

                running_return = 0.0
                for t in reversed(range(episode_length)):
                    if dones[t]:
                        running_return = 0.0
                    running_return = rewards[t] + gamma * running_return
                    discounted_rewards[t] = running_return

                discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-5)

                optimizer_state, policy_network_params, loss = train_step(
                    optimizer_state,
                    policy_network_params,
                    (
                        jnp.asarray(states),
                        jnp.asarray(actions, dtype=jnp.int32),
                        jnp.asarray(discounted_rewards, dtype=jnp.float32)
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
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0
            print(f"Episode {episode}: Average shaped reward over last 10 episodes = {avg_reward:.2f}, Potential Scale Factor: {potential_scale_factor:.3f}, Epsilon: {epsilon:.3f}")

finally:
    env.close()
