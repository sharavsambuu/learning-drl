import random
import math
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


debug_render            = True
debug                   = True
num_episodes            = 5000
learning_rate           = 0.001
gamma                   = 0.99     # Discount factor
sparse_reward_threshold = 50       # Minimum steps to get a reward
sparse_reward_value     = 10       # Reward value for successful episodes


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
        sparse_reward       = 0
        done                = terminated or truncated
        # Only reward if truly terminated (not truncated due to max steps)
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
        logits_list                 = pg_module.apply({'params': params}, batch[0])
        log_probs_list              = jax.nn.log_softmax(logits_list, axis=-1)
        picked_log_probabilities    = gather(log_probs_list, batch[1])
        losses                      = jnp.multiply(picked_log_probabilities, batch[2])
        return -jnp.sum(losses)
    loss, gradients = jax.value_and_grad(loss_fn)(policy_network_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, policy_network_params)
    new_policy_network_params = optax.apply_updates(policy_network_params, updates)
    return new_optimizer_state, new_policy_network_params, loss


global_steps    = 0
episode_rewards = []  # Store rewards for each episode
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
            action = np.random.choice(n_actions, p=probs)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done            = terminated or truncated
            new_state       = np.array(new_state, dtype=np.float32)
            episode_reward += reward

            states .append(state )
            actions.append(action)
            rewards.append(reward)
            dones  .append(done  )

            state = new_state

            if debug_render:
                env.render()

            if done:
                episode_rewards.append(episode_reward)

                if debug:
                    if episode_reward > 0:
                        print(f"{episode} - Sparse Reward Success! Total reward: {episode_reward}, Steps: {len(rewards)}")
                    else:
                        print(f"{episode} - Sparse Reward Failure. Total reward: {episode_reward}, Steps: {len(rewards)}")

                # in sparse setting, skip training if it is zero reward (otherwise gradients are mostly useless)
                if episode_reward > 0:
                    episode_length     = len(rewards)
                    discounted_rewards = np.zeros(episode_length, dtype=np.float32)

                    # O(T) discounted returns (reverse)
                    running_return = 0.0
                    for t in reversed(range(episode_length)):
                        if dones[t]:
                            running_return = 0.0
                        running_return = rewards[t] + gamma * running_return
                        discounted_rewards[t] = running_return

                    discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                    discounted_rewards = discounted_rewards / (np.std(discounted_rewards) + 1e-5)

                    optimizer_state, policy_network_params, loss = train_step(
                        optimizer_state,
                        policy_network_params,
                        (
                            jnp.asarray(states                               ),
                            jnp.asarray(actions           , dtype=jnp.int32  ),
                            jnp.asarray(discounted_rewards, dtype=jnp.float32)
                        )
                    )

                    if debug:
                        print(f"Episode {episode}: Loss = {loss:.4f}")
                else:
                    if debug:
                        print(f"Episode {episode}: skip training (zero sparse reward)")

                break

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 1 else 0.0
            print(f"Episode {episode}: Average reward over last 10 episodes = {avg_reward:.2f}")

finally:
    env.close()
