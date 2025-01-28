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

import os
import random
import math
import gym
from collections import deque

import flax
import jax
from jax import numpy as jnp
import numpy as np

debug_render    = False 
debug           = False
num_episodes    = 500
batch_size      = 64
learning_rate   = 0.001
sync_steps      = 100
memory_length   = 4000
n_step_learning = 3      # N-step learning (Rainbow component)
n_atoms         = 51     # Distributional DQN (C51) atoms (Rainbow component)
v_min           = -10.0  # Distributional DQN V_min
v_max           = 10.0   # Distributional DQN V_max
noisy_std_init  = 0.1    # Noisy Nets std_init (Rainbow component)
epsilon         = 1.0
epsilon_decay   = 0.001
epsilon_max     = 1.0
epsilon_min     = 0.01
gamma           = 0.99   # discount factor


class SumTree: # PER Memory 
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2*capacity - 1)
        self.data     = np.zeros(capacity, dtype=object)
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
    def update(self, idx, p):
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx     = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PERMemory: # PER Memory 
    e = 0.01
    a = 0.6
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    def _get_priority(self, error):
        return (error+self.e)**self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
    def sample(self, n):
        batch   = []
        segment = self.tree.total()/n
        for i in range(n):
            a = segment*i
            b = segment*(i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


def sigma_initializer(value, dtype=jnp.float32): # Noisy Networks initializer (same as Noisy DQN code)
    def init(key, shape, dtype=dtype):
        return jnp.full(shape, value, dtype=dtype)
    return init

class NoisyDense(flax.nn.Module): # Noisy Dense Layer (same as Noisy DQN code)
    def apply(self, x, noise_rng,
            features,
            sigma_init         = 0.017,
            use_bias           = True,
            kernel_initializer = jax.nn.initializers.orthogonal(),
            bias_initializer   = jax.nn.initializers.zeros,
            ):
        input_features = x.shape[-1]
        kernel_shape   = (input_features, features)
        kernel         = self.param('kernel'      , kernel_shape, kernel_initializer)
        sigma_kernel   = self.param('sigma_kernel', kernel_shape, sigma_initializer(value=sigma_init))
        perturbed_kernel = jnp.add(
                kernel,
                jnp.multiply(
                    sigma_kernel,
                    jax.random.uniform(noise_rng, kernel_shape)
                    )
                )
        outputs = jnp.dot(x, perturbed_kernel)
        if use_bias:
            bias       = self.param('bias'      , (features,), bias_initializer)
            sigma_bias = self.param('sigma_bias', (features,), sigma_initializer(value=sigma_init))
            perturbed_bias = jnp.add(
                    bias,
                    jnp.multiply(
                        sigma_bias,
                        jax.random.uniform(noise_rng, (features,))
                        )
                    )
            outputs = jnp.add(outputs, perturbed_bias)
        return outputs

class RainbowDQNNetwork(flax.nn.Module): # Rainbow DQN Network - Combines Dueling, Noisy, Distributional
    def apply(self, x, noise_rng, n_actions, n_atoms): # Takes noise_rng as input for Noisy Layers
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        noisy_layer        = NoisyDense(activation_layer_1, noise_rng, 64) # Noisy Layer
        activation_layer_2 = flax.nn.relu(noisy_layer)

        noisy_value       = NoisyDense(activation_layer_2, noise_rng, 64) # Noisy Layer for Value stream
        value             = flax.nn.relu(noisy_value)
        value             = flax.nn.Dense(value, 1) # Value stream output (scalar)

        noisy_advantage   = NoisyDense(activation_layer_2, noise_rng, 64) # Noisy Layer for Advantage stream
        advantage         = flax.nn.relu(noisy_advantage)
        advantage         = flax.nn.Dense(advantage, n_actions) # Advantage stream output (n_actions,)

        advantage_average = jnp.mean(advantage, keepdims=True) # Dueling layer - Average Dueling
        dueling_output    = jnp.subtract(jnp.add(advantage, value), advantage_average) # Dueling layer - Combine Value and Advantage

        outputs = [] # Distributional output - List of atoms for each action
        for _ in range(n_actions):
            atom_layer      = flax.nn.Dense(dueling_output, n_atoms) # Distributional output layer
            atom_activation = flax.nn.softmax(atom_layer) # Distributional output - Softmax for probabilities
            outputs.append(atom_activation)
        return outputs



env   = gym.make('CartPole-v1')
state = env.reset()

n_actions        = env.action_space.n
dz             = float(v_max-v_min)/(n_atoms-1) # Atom interval (same as C51)
z_holder       = jnp.array([v_min + i*dz for i in range(n_atoms)]) # Atom values (same as C51)


rainbow_dqn_module = RainbowDQNNetwork.partial(n_actions=n_actions, n_atoms=n_atoms) # Use RainbowDQNNetwork
_, params          = rainbow_dqn_module.init_by_shape(
    jax.random.PRNGKey(0),
    [state.shape],
    noise_rng=jax.random.PRNGKey(0) # Dummy noise_rng for initialization
)
q_network          = flax.nn.Model(rainbow_dqn_module, params)
target_q_network   = flax.nn.Model(rainbow_dqn_module, params) # Target network is also RainbowDQNNetwork

optimizer          = flax.optim.Adam(learning_rate).create(q_network)

per_memory         = PERMemory(memory_length)

rng = jax.random.PRNGKey(0) # Random key

@jax.jit
def policy(model, x, rng): # Policy function - Noisy Networks for exploration (no epsilon-greedy)
    rng, noise_rng = jax.random.split(rng) # Split key for noisy layers
    predicted_q_values = model(x, noise_rng=noise_rng) # Pass noise_rng to network
    z_concat = jnp.vstack(predicted_q_values) # Stack atom distributions
    q        = jnp.sum(jnp.multiply(z_concat, z_holder), axis=1) # Expected Q-values (Distributional)
    max_q_action       = jnp.argmax(q) # Choose action with max expected Q-value
    return max_q_action, predicted_q_values, rng # Return rng for stateful policy


@jax.vmap # vmap for TD error calculation (Rainbow TD error)
def calculate_td_error(q_value_vec, next_q_value_vec, target_q_dist_vec, action, reward, discount_factor): # Added target_q_dist_vec for DoubleDQN target action selection
    # Double DQN action selection - use online network for action selection, target network for evaluation
    double_dqn_action = jnp.argmax(next_q_value_vec) # Action selection using online network (Double DQN)
    target_dist       = target_q_dist_vec[double_dqn_action] # Target distribution for selected action (Double DQN)
    td_target         = reward + discount_factor*jnp.sum(target_dist*z_holder) # Distributional N-step TD target (C51 + N-step)
    q_acted           = q_value_vec[action] # Q-value (atom distribution) for taken action
    td_error          = td_target - jnp.sum(q_acted*z_holder) # TD error based on expected Q-values
    return jnp.abs(td_error)

@jax.jit
def td_error(model, online_model, target_model, batch, rng): # TD error function (Rainbow TD error)
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - n_step_rewards
    # batch[3] - next_states
    # batch[4] - dones
    # batch[5] - discount_factors
    rng, noise_rng1, noise_rng2 = jax.random.split(rng, 3)                     # Split keys for noisy networks
    predicted_q_values          = online_model(batch[0], noise_rng=noise_rng1) # Online network for action values
    predicted_next_q_values     = online_model(batch[3], noise_rng=noise_rng2) # Online network for *next* action values (Double DQN action selection)
    target_q_distributions      = target_model(batch[3], noise_rng=noise_rng2) # Target network for target distribution (Double DQN evaluation)
    return calculate_td_error(predicted_q_values, predicted_next_q_values, target_q_distributions, batch[1], batch[2], batch[5]) # Use Rainbow TD error


@jax.vmap # vmap for C51 loss (Categorical Cross-Entropy) (same as C51 code)
def categorical_cross_entropy(predicted_atoms, label_atoms):
    return -jnp.sum(jnp.multiply(label_atoms, jnp.log(predicted_atoms + 1e-8))) # Added small epsilon for numerical stability


@jax.jit
def train_step(optimizer, target_model, batch, rng): # Train step function (Rainbow train step - C51 loss, Double DQN targets)
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - n_step_rewards
    # batch[3] - next_states
    # batch[4] - dones
    # batch[5] - discount_factors
    def loss_fn(model):
        rng_sample, noise_rng = jax.random.split(rng) # Split key for noisy network
        predicted_distributions = model(batch[0], noise_rng=noise_rng) # Online network for predicted distributions

        rng_actions = jax.random.split(rng_sample, batch_size) # Split key for vmap over batch
        target_distributions  = jax.vmap(target_distribution, in_axes=(0,0,0,0,0,None))( # Calculate target distributions using vmap
            batch[3], # next_states
            batch[4], # dones
            batch[2], # n_step_rewards
            batch[5], # discount_factors
            target_model, # target_model
            rng_actions # rng_actions for stochasticity if any in target calculation
        )

        return jnp.mean( # Mean loss over batch
            categorical_cross_entropy(
                jnp.vstack(jax.vmap(lambda i: predicted_distributions[i, batch[1][i]])(jnp.arange(batch_size))), # Predicted distributions for taken actions
                jnp.vstack(target_distributions) # Target distributions
            ))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error(optimizer.target, target_model, batch, rng) # Use Rainbow TD error


@jax.jit # Target distribution calculation for C51 (same as C51 code, but adjusted for N-step reward and discount)
def target_distribution(next_state, done, n_step_reward, discount_factor, target_model, rng):
    rng, noise_rng = jax.random.split(rng) # Split key for noisy network
    target_net_output = target_model(next_state, noise_rng=noise_rng) # Target network for next state distribution
    if done: # Terminal state
        Tz = jnp.clip(n_step_reward, v_min, v_max) # Clip reward to atom range
        b  = (Tz - v_min) / dz # projection index
        lower_bound = jnp.int32(jnp.floor(b))
        upper_bound = jnp.int32(jnp.ceil(b))
        m_prob      = jnp.zeros((n_atoms,))
        m_prob      = m_prob.at[lower_bound].set(m_prob[lower_bound] + (upper_bound - b))
        m_prob      = m_prob.at[upper_bound].set(m_prob[upper_bound] + (b - lower_bound))
        return m_prob
    else: # Non-terminal state
        next_action_dist   = jnp.vstack(target_net_output) # Target network output distributions for all actions
        q_values           = jnp.sum(next_action_dist*z_holder, axis=1) # Expected Q-values for next state
        best_action        = jnp.argmax(q_values) # Best action in next state (Double DQN)
        next_dist          = next_action_dist[best_action] # Distribution of best action in next state
        m_prob             = jnp.zeros((n_atoms,))
        for j in range(n_atoms): # Projection for each atom
            Tz = jnp.clip(n_step_reward + discount_factor*z_holder[j], v_min, v_max) # N-step distributional projection
            b  = (Tz - v_min) / dz
            lower_bound = jnp.int32(jnp.floor(b))
            upper_bound = jnp.int32(jnp.ceil(b))
            m_prob      = m_prob.at[lower_bound].set(m_prob[lower_bound] + next_dist[j] * (upper_bound - b))
            m_prob      = m_prob.at[upper_bound].set(m_prob[upper_bound] + next_dist[j] * (b - lower_bound))
        return m_prob


fig = plt.gcf() # Visualization setup (same as C51/QR-DQN - optional)
fig.show()
fig.canvas.draw()


rng      = jax.random.PRNGKey(0) # Random key
global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        n_step_buffer = deque(maxlen=n_step_learning) # N-step buffer (same as N-step DQN)
        while True:
            global_steps = global_steps+1
            rng, policy_key, train_key = jax.random.split(rng, 3) # Split keys

            action, q_values, rng = policy(optimizer.target, jnp.asarray([state]), rng=policy_key) # Policy call (Noisy Nets exploration)
            if debug_render:
                plt.clf() # Visualization (optional)
                # Visualization will be different for Rainbow DQN, customize as needed
                pass
            action = np.array(action)


            if epsilon>epsilon_min: # Epsilon-greedy (can be removed if Noisy Nets exploration is sufficient)
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()


            new_state, reward, done, _ = env.step(int(action))

            # Store step in N-step buffer (same as N-step DQN)
            n_step_buffer.append((state, action, reward, new_state, done))

            # When N-step buffer is full, or episode ends, calculate N-step return and add to PER memory (same as N-step DQN)
            if len(n_step_buffer) == n_step_learning or done:
                n_step_state, n_step_action, n_step_reward_list, n_step_next_state, n_step_done = n_step_buffer[0]
                cumulative_reward = 0
                discount_factor   = 1.0
                for i in range(len(n_step_buffer)):
                    cumulative_reward += discount_factor * n_step_buffer[i][2]
                    discount_factor   *= gamma
                n_step_reward     = cumulative_reward

                # sample нэмэхдээ temporal difference error-ийг тооцож нэмэх (using Rainbow TD error)
                temporal_difference = float(td_error(optimizer.target, q_network, target_q_network, ( # Pass online_model as q_network for DoubleDQN
                        jnp.asarray([n_step_state]),    # N-step start state
                        jnp.asarray([n_step_action]),   # N-step action
                        jnp.asarray([n_step_reward]),   # N-step reward
                        jnp.asarray([new_state]),       # N-step end state (current new_state)
                        jnp.asarray([int(done)]),       # N-step done
                        jnp.asarray([discount_factor])  # N-step discount factor
                    ), train_key)[0])
                per_memory.add(temporal_difference, (n_step_state, n_step_action, n_step_reward, new_state, int(done), discount_factor)) # Store N-step transition

            if len(per_memory) > batch_size:
                # Prioritized Experience Replay санах ойгоос batch үүсгээд DQN сүлжээг сургах (using Rainbow DQN)
                batch = per_memory.sample(batch_size)
                states, actions, n_step_rewards, next_states, dones, discount_factors = [], [], [], [], [], []
                for i in range(batch_size):
                    states.append        (batch[i][1][0])
                    actions.append       (batch[i][1][1])
                    n_step_rewards.append(batch[i][1][2])
                    next_states.append   (batch[i][1][3])
                    dones.append         (batch[i][1][4])
                    discount_factors.append(batch[i][1][5])


                rng, train_key = jax.random.split(rng, 2) # Split key for train step
                optimizer, loss, new_td_errors = train_step( # Train step using Rainbow train_step
                                            optimizer,
                                            target_q_network,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур
                                                # төхөөрөмжийн санах ойруу хуулах (Rainbow DQN batch)
                                                jnp.asarray(states),
                                                jnp.asarray(actions),
                                                jnp.asarray(n_step_rewards), # N-step rewards
                                                jnp.asarray(next_states),   # N-step next states
                                                jnp.asarray(dones),
                                                jnp.asarray(discount_factors) # N-step discount factors
                                            ),
                                            train_key # Pass key to train step
                                        )
                # batch-аас бий болсон temporal difference error-ийн дагуу санах ойг шинэчлэх (using Rainbow TD error)
                new_td_errors = np.array(new_td_errors)
                for i in range(batch_size):
                    idx = batch[i][0]
                    per_memory.update(idx, new_td_errors[i])


            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0:
                target_q_network = target_q_network.replace(params=optimizer.target.params)
                if debug:
                    print("сайжруулсан жингүүдийг target неорон сүлжээрүү хууллаа")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
            if len(n_step_buffer) == n_step_learning: # Clear N-step buffer (same as N-step DQN)
                n_step_buffer.popleft()
finally:
    env.close()