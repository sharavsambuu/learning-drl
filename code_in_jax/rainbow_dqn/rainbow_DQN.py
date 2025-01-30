#
#
# Rainbow DQN integrates these six key DQN enhancements into a single agent:
#
#   - Double DQN: Addresses the overestimation bias in Q-Learning by decoupling action selection and evaluation in TD target calculation. 
#   - Prioritized Experience Replay (PER): Prioritizes the replay of important transitions (high TD-error) for more efficient learning. 
#   - Dueling Networks: Separates the value and advantage streams in the network architecture for more efficient learning of state values and action advantages. 
#   - Noisy Networks: Replaces deterministic layers with noisy layers for more efficient exploration, removing the need for epsilon-greedy. 
#   - Distributional DQN (C51): Learns the distribution of returns instead of just the mean Q-value, providing a richer representation of value. 
#   - N-step Learning: Uses N-step returns to bootstrap over multiple steps, balancing bias and variance in TD learning. 
#
#
#   Rainbow DQN = DQN + Double DQN + PER + Dueling Nets + Noisy Nets + C51 + N-step Learning
#
#
#

import random
import gymnasium as gym
from collections import deque
import flax.linen as nn
from flax.linen import initializers
import jax
from jax import numpy as jnp
import numpy as np
import optax

debug_render  = True
debug         = False
num_episodes  = 500
batch_size    = 64
learning_rate = 0.00025
sync_steps    = 100
memory_length = 10000
n_steps       = 3

gamma         = 0.99

# C51 parameters
v_min         = -10.0
v_max         = 10.0
n_atoms       = 51
z_holder      = jnp.linspace(v_min, v_max, n_atoms)
dz            = (v_max - v_min) / (n_atoms - 1)

# PER parameters
per_e         = 0.01
per_a         = 0.6
per_b_start   = 0.4
per_b_end     = 1.0
per_b_decay   = num_episodes
beta          = per_b_start

# Noisy Nets
sigma_init    = 0.5

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2*capacity - 1)
        self.data     = [None] * capacity
        self.size     = 0
    def _propagate(self, idx, change):
        parent             = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])
    def total(self):
        return self.tree[0]
    def add(self, p, data):
        idx                   = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write           += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1
    def update(self, idx, p):
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx     = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PERMemory:
    e = per_e
    a = per_a
    beta_start = per_b_start
    beta_end   = per_b_end
    beta_decay_episodes = per_b_decay
    beta       = per_b_start
    memory_count = 0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (error+self.e)**self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.memory_count = min(self.memory_count + 1, self.capacity)

    def sample(self, n, episode_num):
        batch       = []
        idxs        = []
        is_weights  = []
        segment     = self.tree.total()/n
        self.beta = min(self.beta_end, self.beta_start + episode_num * (self.beta_end - self.beta_start) / self.beta_decay_episodes)

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total() if self.tree.total() > 0 else 1e-8
        max_weight = (max(min_prob, 1e-8) * self.memory_count) ** (-self.beta)

        for i in range(n):
            a = segment*i
            b = segment*(i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            if data is None:
                continue

            batch.append(data)
            idxs.append(idx)

            prob = p / self.tree.total() if self.tree.total() > 0 else 1e-8
            weight = (prob * self.memory_count) ** (-self.beta)
            is_weights.append(weight / max_weight if max_weight > 0 else 1.0)

        valid_batch_indices = [i for i, data in enumerate(batch) if data is not None]
        valid_batch = [batch[i] for i in valid_batch_indices]
        valid_idxs = [idxs[i] for i in valid_batch_indices]
        valid_is_weights = [is_weights[i] for i in valid_batch_indices]

        if not valid_batch:
            return None, None, None

        return valid_batch, valid_idxs, jnp.array(valid_is_weights, dtype=jnp.float32)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

def uniform(scale=0.05, dtype=jnp.float_):
  def init(key, shape, dtype=dtype):
    return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
  return init

class NoisyDense(nn.Module):
    features   : int
    use_bias   : bool                     = True
    sigma_init : float                    = sigma_init
    kernel_init: initializers.Initializer = uniform()
    bias_init  : initializers.Initializer = initializers.zeros

    @nn.compact
    def __call__(self, inputs, noise_key):
        input_shape  = inputs.shape[-1]
        kernel       = self.param('kernel', self.kernel_init, (input_shape, self.features))
        sigma_kernel = self.param(
            'sigma_kernel',
            uniform(scale=self.sigma_init),
            (input_shape, self.features),
        )
        kernel_noise     = jax.random.normal(noise_key, (input_shape, self.features))
        perturbed_kernel = kernel + sigma_kernel * kernel_noise

        outputs = jnp.dot(inputs, perturbed_kernel)

        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
            sigma_bias = self.param(
                'sigma_bias',
                uniform(scale=self.sigma_init),
                (self.features,),
            )
            bias_noise = jax.random.normal(jax.random.fold_in(noise_key, 1), (self.features,))
            perturbed_bias = bias + sigma_bias * bias_noise
            outputs = outputs + perturbed_bias
        return outputs

class NoisyDuelingQNetwork(nn.Module):
    n_actions: int
    n_atoms:   int

    @nn.compact
    def __call__(self, x, noise_key):
        noisy_dense_key, value_key, advantage_key = jax.random.split(noise_key, num=3)

        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = NoisyDense(features=64)(x, noisy_dense_key)
        x = nn.relu(x)

        # Value stream
        value = NoisyDense(features=64)(x, noisy_dense_key)
        value = nn.relu(value)
        value = NoisyDense(features=self.n_atoms)(value, value_key)

        # Advantage stream
        advantage = NoisyDense(features=64)(x, noisy_dense_key)
        advantage = nn.relu(advantage)
        advantage = NoisyDense(features=self.n_actions * self.n_atoms)(advantage, advantage_key)

        # Reshape to distributions
        value_distributions     = value.reshape((-1, 1, self.n_atoms))
        advantage_distributions = advantage.reshape((-1, self.n_actions, self.n_atoms))

        # Combine streams for distributions
        q_distributions = value_distributions + (advantage_distributions - jnp.mean(advantage_distributions, axis=1, keepdims=True))
        return q_distributions

env         = gym.make('CartPole-v1', render_mode="human")
state, info = env.reset()
state       = np.array(state, dtype=np.float32)

n_actions   = env.action_space.n

# Initialize the Noisy Dueling Q-Network and the target network
dqn_module       = NoisyDuelingQNetwork(n_actions=n_actions, n_atoms=n_atoms)
dummy_input      = jnp.zeros(state.shape)
params           = dqn_module.init(
    {'params':jax.random.PRNGKey(0)},
    dummy_input,
    noise_key=jax.random.PRNGKey(0))
q_network_params = params['params']
target_q_network_params = params['params']

optimizer        = optax.adam(learning_rate)
opt_state        = optimizer.init(q_network_params)

per_memory       = PERMemory(memory_length)
n_step_memory    = deque(maxlen=n_steps)

@jax.jit
def policy(params, x, noise_key):
    q_distributions    = dqn_module.apply({'params': params}, x, noise_key=noise_key)
    q_values           = jnp.sum(q_distributions * z_holder, axis=2)
    max_q_action       = jnp.argmax(q_values)
    return max_q_action, q_values

# Projection Distribution for C51
def projection_distribution(next_q_dist, rewards, dones):
    batch_size      = next_q_dist.shape[0]
    projected_dist  = jnp.zeros((batch_size, n_atoms))
    argmax_actions  = jnp.argmax(jnp.sum(next_q_dist * z_holder, axis=2), axis=1)

    for sample_idx in range(batch_size):
        is_done = dones[sample_idx] > 0

        def done_case():
            tz = jnp.clip(rewards[sample_idx], v_min, v_max)
            bj = (tz - v_min) / dz
            l, u = jnp.floor(bj).astype(jnp.int32), jnp.ceil(bj).astype(jnp.int32)
            dist = jnp.zeros(n_atoms)
            dist = dist.at[l].add(1.0)
            dist = jnp.where(u != l, dist.at[u].add(1.0), dist)
            return dist

        def not_done_case():
            dist = jnp.zeros(n_atoms)
            for atom_idx in range(n_atoms):
                tz = jnp.clip(rewards[sample_idx] + gamma * z_holder[atom_idx], v_min, v_max)
                bj = (tz - v_min) / dz
                l, u = jnp.floor(bj).astype(jnp.int32), jnp.ceil(bj).astype(jnp.int32)
                prob = next_q_dist[sample_idx, argmax_actions[sample_idx], atom_idx]
                dist = dist.at[l].add(prob * (u - bj))
                dist = jnp.where(u != l, dist.at[u].add(prob * (bj - l)), dist)
            return dist

        projected_dist = projected_dist.at[sample_idx].set(jnp.where(is_done, done_case(), not_done_case()))
    return projected_dist

# Categorical Cross-Entropy Loss for C51
def categorical_loss(q_distributions, target_distributions, actions):
    log_prob_actions = jax.nn.log_softmax(q_distributions[jnp.arange(batch_size), actions], axis=-1)
    loss = -jnp.sum(target_distributions * log_prob_actions, axis=-1)
    return loss

@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch, noise_key, is_weights):
    states, actions, rewards, next_states, dones = batch

    def loss_fn(params, noise_key):
        noise_sample_key, noise_loss_key = jax.random.split(noise_key)

        target_noise_key, _ = jax.random.split(noise_sample_key)
        target_q_dist = dqn_module.apply({'params': target_q_network_params}, next_states, noise_key=target_noise_key)

        projected_distributions = projection_distribution(target_q_dist, rewards, dones)

        online_q_dist = dqn_module.apply({'params': params}, states, noise_key=noise_loss_key)
        losses = categorical_loss(online_q_dist, projected_distributions, actions)
        weighted_loss = losses * is_weights
        return jnp.mean(weighted_loss), losses

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, td_errors), gradients = grad_fn(q_network_params, noise_key)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params = optax.apply_updates(q_network_params, updates)
    return q_network_params, opt_state, loss, td_errors

rng = jax.random.PRNGKey(0)
noise_key = jax.random.PRNGKey(0)

global_steps = 0
episode_rewards_history = deque(maxlen=100)
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        while True:
            global_steps = global_steps+1

            rng, key = jax.random.split(rng)
            noise_key, n_key = jax.random.split(noise_key)
            action, q_values = policy(q_network_params, state, noise_key=n_key)
            action = int(action)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            n_step_memory.append((state, action, reward, new_state, float(done)))

            # N-step Learning & PER Buffer Add
            if len(n_step_memory) == n_steps or done:
                state_0, action_0, reward_0, next_state_n_step, done_n_step = n_step_memory.popleft()
                n_step_reward = reward_0
                discount_factor = gamma
                for i in range(len(n_step_memory)):
                    n_step_reward += n_step_memory[i][2] * (discount_factor**(i+1))

                initial_priority_error = 1.0

                per_memory.add(
                    initial_priority_error,
                    (state_0, action_0, n_step_reward, next_state_n_step, float(done_n_step))
                )

            if (per_memory.tree.size>batch_size and global_steps % 1 == 0):
                batch, idxs, is_weights = per_memory.sample(batch_size, episode)

                if batch is not None:
                    states, actions, rewards, next_states, dones = zip(*batch)

                    noise_key, n_key = jax.random.split(noise_key)
                    q_network_params, opt_state, loss, new_td_errors = train_step(
                        q_network_params,
                        target_q_network_params,
                        opt_state,
                        (
                            jnp.asarray(list(states)),
                            jnp.asarray(list(actions), dtype=jnp.int32),
                            jnp.asarray(list(rewards), dtype=jnp.float32),
                            jnp.asarray(list(next_states)),
                            jnp.asarray(list(dones), dtype=jnp.float32)
                        ),
                        noise_key=n_key,
                        is_weights=is_weights
                    )
                    new_td_errors_np = np.array(new_td_errors)
                    for i in range(len(idxs)):
                        idx = idxs[i]
                        per_memory.update(idx, new_td_errors_np[i])

            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0:
                target_q_network_params = q_network_params
                print("copied updated weights to the target network")

            if debug_render:
                env.render()

            if done:
                episode_reward_sum = sum(episode_rewards)
                episode_rewards_history.append(episode_reward_sum)
                avg_reward = np.mean(episode_rewards_history)
                print(f"Episode {episode+1}, Global Steps: {global_steps}, Reward: {episode_reward_sum:.2f}, Avg Reward (Last 100): {avg_reward:.2f}, Beta: {per_memory.beta:.2f}")
                break
finally:
    env.close()