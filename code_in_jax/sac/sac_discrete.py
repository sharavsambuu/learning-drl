import random
import jax
import optax
import gymnasium as gym
import flax.linen as nn
import numpy as np
from jax import numpy as jnp

num_episodes    = 500
learning_rate   = 0.001
gamma           = 0.99
tau             = 0.005  # Soft update coefficient for target network
entropy_alpha   = 0.2    # Initial entropy temperature (can be tuned or made learnable)
batch_size      = 64
memory_length   = 10000

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
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity
    def _get_priority(self, error):
        return (np.abs(error) + per_epsilon) ** self.alpha
    def push(self, state, action, reward, next_state, done):
        error = 1.0  # Initial error for new transitions
        p = self._get_priority(error)
        self.tree.add(p, (state, action, reward, next_state, done))
    def sample(self, batch_size, beta):
        batch = []
        idxs  = []
        segment = self.tree.total() / batch_size
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
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x

class DiscretePolicyNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        x = nn.softmax(x)
        return x


#env         = gym.make('CartPole-v1', render_mode='human')
env         = gym.make('LunarLander-v3', render_mode='human')
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n

soft_q_module_1 = SoftQNetwork(n_actions=n_actions)
soft_q_module_2 = SoftQNetwork(n_actions=n_actions)
policy_module   = DiscretePolicyNetwork(n_actions=n_actions)
dummy_input     = jnp.zeros(state.shape)

soft_q_params_1        = soft_q_module_1.init(jax.random.PRNGKey(0), dummy_input)['params']
soft_q_params_2        = soft_q_module_2.init(jax.random.PRNGKey(1), dummy_input)['params']
target_soft_q_params_1 = soft_q_params_1
target_soft_q_params_2 = soft_q_params_2
policy_params          = policy_module.init(jax.random.PRNGKey(2), dummy_input)['params']

soft_q_optimizer_1 = optax.adam(learning_rate)
soft_q_optimizer_2 = optax.adam(learning_rate)
policy_optimizer   = optax.adam(learning_rate)
soft_q_opt_state_1 = soft_q_optimizer_1.init(soft_q_params_1)
soft_q_opt_state_2 = soft_q_optimizer_2.init(soft_q_params_2)
policy_opt_state   = policy_optimizer.init(policy_params)


replay_memory = PrioritizedReplayMemory(memory_length)


@jax.jit
def soft_q_inference(params, x):
    return soft_q_module_1.apply({'params': params}, x)

@jax.jit
def policy_inference(params, x):
    return policy_module.apply({'params': params}, x)

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def soft_q_update(
    soft_q_params_1, soft_q_opt_state_1,
    soft_q_params_2,
    target_soft_q_params_1, target_soft_q_params_2,
    policy_params,
    states, actions, rewards, next_states, dones, is_weights
):
    # Calculate target Q-value
    next_state_policy_probs = policy_module.apply({'params': policy_params}, next_states)
    next_state_q_values_1 = soft_q_module_1.apply({'params': target_soft_q_params_1}, next_states)
    next_state_q_values_2 = soft_q_module_2.apply({'params': target_soft_q_params_2}, next_states)
    next_state_min_q = jnp.minimum(next_state_q_values_1, next_state_q_values_2)
    
    next_state_v = jnp.sum(next_state_policy_probs * (next_state_min_q - entropy_alpha * jnp.log(next_state_policy_probs + 1e-6)), axis=1)
    
    target_q_values = rewards + (1.0 - dones) * gamma * next_state_v

    # Calculate Q-function loss with Importance Sampling weights
    def soft_q_loss_fn(params, target, actions, q_net_module, is_weights):
        q_values = q_net_module.apply({'params': params}, states)
        q_values_of_actions = gather(q_values, actions)
        td_error = target - q_values_of_actions
        return jnp.mean(is_weights * jnp.square(td_error)), td_error

    # Update Q-function 1
    (q1_loss, q1_td_error), q1_grads = jax.value_and_grad(soft_q_loss_fn, has_aux=True)(soft_q_params_1, target_q_values, actions, soft_q_module_1, is_weights)
    q1_updates, new_soft_q_opt_state_1 = soft_q_optimizer_1.update(q1_grads, soft_q_opt_state_1, soft_q_params_1)
    new_soft_q_params_1 = optax.apply_updates(soft_q_params_1, q1_updates)

    # Update Q-function 2
    (q2_loss, q2_td_error), q2_grads = jax.value_and_grad(soft_q_loss_fn, has_aux=True)(soft_q_params_2, target_q_values, actions, soft_q_module_2, is_weights)
    q2_updates, new_soft_q_opt_state_2 = soft_q_optimizer_2.update(q2_grads, soft_q_opt_state_2, soft_q_params_2)
    new_soft_q_params_2 = optax.apply_updates(soft_q_params_2, q2_updates)

    return new_soft_q_params_1, new_soft_q_opt_state_1, new_soft_q_params_2, new_soft_q_opt_state_2, q1_loss, q2_loss, q1_td_error, q2_td_error

@jax.jit
def policy_update(policy_params, policy_opt_state, soft_q_params_1, soft_q_params_2, states):
    def policy_loss_fn(params):
        policy_probs = policy_module.apply({'params': params}, states)
        
        soft_q_values_1 = soft_q_module_1.apply({'params': soft_q_params_1}, states)
        soft_q_values_2 = soft_q_module_2.apply({'params': soft_q_params_2}, states)
        min_soft_q_values = jnp.minimum(soft_q_values_1, soft_q_values_2)

        log_probs = min_soft_q_values - jax.scipy.special.logsumexp(min_soft_q_values, axis=1, keepdims=True)
        
        policy_loss = jnp.sum(policy_probs * (entropy_alpha * jnp.log(policy_probs + 1e-6) - min_soft_q_values), axis=1)
        return jnp.mean(policy_loss)

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
            if np.random.rand() < 0.1:
                action = env.action_space.sample()
            else:
                action_probs = policy_inference(policy_params, jnp.array([state]))
                action = int(jnp.argmax(action_probs))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward

            replay_memory.push(state, action, reward, next_state, float(done))

            # Linearly annealing beta from per_beta_start to 1.0
            per_beta = min(1.0, per_beta_start + global_step * (1.0 - per_beta_start) / per_beta_frames)

            if len(replay_memory) > batch_size:
                idxs, states, actions, rewards, next_states, dones, is_weights = replay_memory.sample(batch_size, per_beta)

                states      = jnp.array(states     )
                actions     = jnp.array(actions    )
                rewards     = jnp.array(rewards    )
                next_states = jnp.array(next_states)
                dones       = jnp.array(dones      )
                is_weights  = jnp.array(is_weights )

                soft_q_params_1, soft_q_opt_state_1, soft_q_params_2, soft_q_opt_state_2, q1_loss, q2_loss, q1_td_error, q2_td_error = soft_q_update(
                    soft_q_params_1,
                    soft_q_opt_state_1,
                    soft_q_params_2,
                    target_soft_q_params_1,
                    target_soft_q_params_2,
                    policy_params,
                    states, actions, rewards, next_states, dones, is_weights
                )

                policy_params, policy_opt_state, policy_loss = policy_update(
                    policy_params,
                    policy_opt_state,
                    soft_q_params_1,
                    soft_q_params_2,
                    states
                )

                # Update priorities in PER tree
                for idx, error in zip(idxs, np.mean((np.abs(q1_td_error), np.abs(q2_td_error)), axis=0)):
                    replay_memory.update(idx, error)

                # Soft update target networks
                target_soft_q_params_1 = soft_update(target_soft_q_params_1, soft_q_params_1, tau)
                target_soft_q_params_2 = soft_update(target_soft_q_params_2, soft_q_params_2, tau)

            state = next_state

        print(f"Episode: {episode}, Total Reward: {total_reward}")

finally:
    env.close()
