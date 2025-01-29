import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import random

num_episodes    = 500
learning_rate   = 0.0003   # Lower learning rate for continuous SAC, often helps
gamma           = 0.99
tau             = 0.005
entropy_alpha   = 0.2
batch_size      = 256      # Larger batch size for continuous SAC
memory_length   = 1000000  # Larger memory for continuous SAC
per_alpha       = 0.6
per_beta_start  = 0.4
per_beta_frames = 100000
per_epsilon     = 1e-6


class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    def total(self):
        return self.tree[0]
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=per_alpha):
        self.tree     = SumTree(capacity)
        self.alpha    = alpha
        self.capacity = capacity
    def _get_priority(self, error):
        return (np.abs(error) + per_epsilon) ** self.alpha
    def push(self, state, action, reward, next_state, done):
        error = 1.0  # Initial error for new transitions
        p = self._get_priority(error)
        self.tree.add(p, (state, action, reward, next_state, done))
    def sample(self, batch_size, beta):
        batch      = []
        idxs       = []
        segment    = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        states, actions, rewards, next_states, dones = zip(*batch)
        return idxs, np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(
            dones), is_weight
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    def __len__(self):
        return min(self.tree.write, self.capacity)


class SoftQNetwork(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)  # Combine state and action
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

class GaussianPolicyNetwork(nn.Module):
    action_dim : int
    log_std_min: float = -20.
    log_std_max: float = 2.
    @nn.compact
    def __call__(self, state):
        x = nn.Dense(features=256)(state)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        mean = nn.Dense(features=self.action_dim)(x)
        log_std = nn.Dense(features=self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

def sample_action(rng, policy_params, policy_module, state, epsilon=1e-6):
    mean, log_std = policy_module.apply({'params': policy_params}, state)
    std = jnp.exp(log_std)
    normal = jax.random.normal(rng, mean.shape)
    z = mean + std * normal
    action = jnp.tanh(z)
    log_prob = gaussian_likelihood(z, mean, log_std) - jnp.log(1 - action ** 2 + epsilon)
    log_prob = log_prob.sum(axis=-1, keepdims=True)
    return action, log_prob

def gaussian_likelihood(noise, mean, log_std):
    pre_sum = -0.5 * (((noise - mean) / (jnp.exp(log_std) + 1e-6)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return pre_sum


# Other envs MountainCarContinuous-v0
env = gym.make('Pendulum-v1', render_mode="human")
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
action_dim  = env.action_space.shape[0]
max_action  = env.action_space.high[0]

soft_q_module_1 = SoftQNetwork()
soft_q_module_2 = SoftQNetwork()
policy_module   = GaussianPolicyNetwork(action_dim=action_dim)

soft_q_params_1        = soft_q_module_1.init(jax.random.PRNGKey(0), jnp.array([state]), jnp.array([[0.0]]))['params']
soft_q_params_2        = soft_q_module_2.init(jax.random.PRNGKey(1), jnp.array([state]), jnp.array([[0.0]]))['params']
target_soft_q_params_1 = soft_q_params_1
target_soft_q_params_2 = soft_q_params_2
policy_params          = policy_module.init(jax.random.PRNGKey(2), jnp.array([state]))['params']

soft_q_optimizer_1 = optax.adam(learning_rate)
soft_q_optimizer_2 = optax.adam(learning_rate)
policy_optimizer   = optax.adam(learning_rate)
soft_q_opt_state_1 = soft_q_optimizer_1.init(soft_q_params_1)
soft_q_opt_state_2 = soft_q_optimizer_2.init(soft_q_params_2)
policy_opt_state   = policy_optimizer.init(policy_params)

replay_memory = PrioritizedReplayMemory(memory_length)


@jax.jit
def soft_q_inference(params, state, action):
    return soft_q_module_1.apply({'params': params}, state, action)

@jax.jit
def policy_inference(params, state):
    return policy_module.apply({'params': params}, state)

@jax.jit
def soft_q_update(
    soft_q_params_1, soft_q_opt_state_1,
    soft_q_params_2, soft_q_opt_state_2,
    target_soft_q_params_1, target_soft_q_params_2,
    policy_params,
    states, actions, rewards, next_states, dones, is_weights
):
    # Sample actions and log probabilities for next states
    rng = jax.random.PRNGKey(0)
    next_actions, next_log_probs = sample_action(rng, policy_params, policy_module, next_states)
    
    # Calculate target Q-value
    target_q_values_1 = soft_q_module_1.apply({'params': target_soft_q_params_1}, next_states, next_actions)
    target_q_values_2 = soft_q_module_2.apply({'params': target_soft_q_params_2}, next_states, next_actions)
    target_q_values = jnp.minimum(target_q_values_1, target_q_values_2) - entropy_alpha * next_log_probs
    target_q_values = rewards + (1.0 - dones) * gamma * target_q_values

    # Calculate Q-function loss with Importance Sampling weights
    def soft_q_loss_fn(params, target, states, actions, is_weights, q_net_module):
        q_values = q_net_module.apply({'params': params}, states, actions)
        td_error = target - q_values
        return jnp.mean(is_weights * jnp.square(td_error)), td_error

    # Update Q-function 1
    (q1_loss, q1_td_error), q1_grads = jax.value_and_grad(soft_q_loss_fn, has_aux=True)(soft_q_params_1, target_q_values, states, actions, is_weights, soft_q_module_1)
    q1_updates, new_soft_q_opt_state_1 = soft_q_optimizer_1.update(q1_grads, soft_q_opt_state_1, soft_q_params_1)
    new_soft_q_params_1 = optax.apply_updates(soft_q_params_1, q1_updates)

    # Update Q-function 2
    (q2_loss, q2_td_error), q2_grads = jax.value_and_grad(soft_q_loss_fn, has_aux=True)(soft_q_params_2, target_q_values, states, actions, is_weights, soft_q_module_2)
    q2_updates, new_soft_q_opt_state_2 = soft_q_optimizer_2.update(q2_grads, soft_q_opt_state_2, soft_q_params_2)
    new_soft_q_params_2 = optax.apply_updates(soft_q_params_2, q2_updates)

    return new_soft_q_params_1, new_soft_q_opt_state_1, new_soft_q_params_2, new_soft_q_opt_state_2, q1_loss, q2_loss, q1_td_error, q2_td_error

@jax.jit
def policy_update(policy_params, policy_opt_state, soft_q_params_1, soft_q_params_2, states, rng):
    def policy_loss_fn(params):
        # Sample action for the current states
        actions, log_probs = sample_action(rng, params, policy_module, states)

        # Calculate Q-values for current states and sampled actions
        q_values_1 = soft_q_module_1.apply({'params': soft_q_params_1}, states, actions)
        q_values_2 = soft_q_module_2.apply({'params': soft_q_params_2}, states, actions)
        min_q_values = jnp.minimum(q_values_1, q_values_2)

        # Calculate policy loss
        policy_loss = jnp.mean(entropy_alpha * log_probs - min_q_values)

        return policy_loss

    (policy_loss, policy_grads) = jax.value_and_grad(policy_loss_fn)(policy_params)
    policy_updates, new_policy_opt_state = policy_optimizer.update(policy_grads, policy_opt_state, policy_params)
    new_policy_params = optax.apply_updates(policy_params, policy_updates)

    return new_policy_params, new_policy_opt_state, policy_loss

@jax.jit
def soft_update(target_params, params, tau):
    def update_fn(target_param, param):
        return (1 - tau) * target_param + tau * param
    return jax.tree_map(update_fn, target_params, params)


global_step = 0
per_beta    = per_beta_start

try:
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0

        while not done:
            global_step += 1
            # Select action
            rng = jax.random.PRNGKey(global_step)
            action, _ = sample_action(rng, policy_params, policy_module, jnp.array([state]))
            action = np.array(action[0])  # Convert to a NumPy array for environment step

            next_state, reward, terminated, truncated, info = env.step(action * max_action)  # Scale action
            done          = terminated or truncated
            next_state    = np.array(next_state, dtype=np.float32)
            total_reward += reward

            replay_memory.push(state, action, reward, next_state, float(done))

            # Linearly annealing beta from per_beta_start to 1.0
            per_beta = min(1.0, per_beta_start + global_step * (1.0 - per_beta_start) / per_beta_frames)

            if len(replay_memory) > batch_size:
                idxs, states, actions, rewards, next_states, dones, is_weights = replay_memory.sample(batch_size, per_beta)

                states      = jnp.array(states)
                actions     = jnp.array(actions)
                rewards     = jnp.array(rewards)
                next_states = jnp.array(next_states)
                dones       = jnp.array(dones)
                is_weights  = jnp.array(is_weights)

                soft_q_params_1, soft_q_opt_state_1, soft_q_params_2, soft_q_opt_state_2, q1_loss, q2_loss, q1_td_error, q2_td_error = soft_q_update(
                    soft_q_params_1,
                    soft_q_opt_state_1,
                    soft_q_params_2,
                    soft_q_opt_state_2,
                    target_soft_q_params_1,
                    target_soft_q_params_2,
                    policy_params,
                    states, actions, rewards, next_states, dones, is_weights
                )

                # Use a new RNG key for policy update
                rng, policy_rng = jax.random.split(rng)
                policy_params, policy_opt_state, policy_loss = policy_update(
                    policy_params,
                    policy_opt_state,
                    soft_q_params_1,
                    soft_q_params_2,
                    states,
                    policy_rng
                )

                # Calculate new priorities
                new_priorities = np.max(np.abs(np.concatenate([q1_td_error, q2_td_error])), axis=0)

                # Update priorities in PER tree
                for idx, priority in zip(idxs, new_priorities):
                    replay_memory.update(idx, float(priority))

                # Soft update target networks
                target_soft_q_params_1 = soft_update(target_soft_q_params_1, soft_q_params_1, tau)
                target_soft_q_params_2 = soft_update(target_soft_q_params_2, soft_q_params_2, tau)

            state = next_state

        print(f"Episode: {episode}, Total Reward: {total_reward}")

finally:
    env.close()