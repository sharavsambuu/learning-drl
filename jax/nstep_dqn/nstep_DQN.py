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
n_step_learning = 3      # N-step parameter: use 3-step returns

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


class DeepQNetwork(flax.nn.Module): # DQN Network 
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_actions)
        return output_layer


env   = gym.make('CartPole-v1')
state = env.reset()

n_actions        = env.action_space.n

dqn_module       = DeepQNetwork.partial(n_actions=n_actions)
_, params        = dqn_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
q_network        = flax.nn.Model(dqn_module, params)
target_q_network = flax.nn.Model(dqn_module, params)

optimizer        = flax.optim.Adam(learning_rate).create(q_network)

per_memory       = PERMemory(memory_length)

# N-step buffer to store N-step transitions
n_step_buffer = deque(maxlen=n_step_learning)

@jax.jit
def policy(model, x): # Policy function 
    predicted_q_values = model(x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


@jax.vmap # vmap for TD error calculation (adjusted for N-step)
def calculate_td_error(q_value_vec, target_q_value_vec, action, n_step_reward, discount_factor):
    td_target = n_step_reward + discount_factor*jnp.max(target_q_value_vec) # N-step TD target
    td_error  = td_target - q_value_vec[action]
    return jnp.abs(td_error)

@jax.jit
def td_error(model, target_model, batch): # TD error function (adjusted for N-step)
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - n_step_rewards (N-step rewards)
    # batch[3] - next_states (N-step next states)
    # batch[4] - discount_factors (N-step discount factors)
    predicted_q_values = model(batch[0])
    target_q_values    = target_model(batch[3])
    return calculate_td_error(predicted_q_values, target_q_values, batch[1], batch[2], batch[4]) # Use N-step reward and discount

@jax.vmap # vmap for Q-learning loss (adjusted for N-step)
def q_learning_loss(q_value_vec, target_q_value_vec, action, n_step_reward, done, discount_factor):
    td_target = n_step_reward + discount_factor*jnp.max(target_q_value_vec)*(1.-done) # N-step TD target
    td_error  = jax.lax.stop_gradient(td_target) - q_value_vec[action]
    return jnp.square(td_error)

@jax.jit
def train_step(optimizer, target_model, batch): # Train step function (adjusted for N-step)
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - n_step_rewards
    # batch[3] - next_states
    # batch[4] - dones
    # batch[5] - discount_factors
    def loss_fn(model):
        predicted_q_values      = model(batch[0])
        target_q_values         = target_model(batch[3])
        return jnp.mean(
                q_learning_loss(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4],
                    batch[5]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error(optimizer.target, target_model, batch) # Use N-step TD error


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        n_step_buffer.clear() # Clear N-step buffer at start of episode
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(optimizer.target, state)
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
                if debug:
                    #print("epsilon :", epsilon)
                    pass

            new_state, reward, done, _ = env.step(int(action))

            # Store step in N-step buffer
            n_step_buffer.append((state, action, reward, new_state, done))

            # When N-step buffer is full, or episode ends, calculate N-step return and add to PER memory
            if len(n_step_buffer) == n_step_learning or done:
                # Calculate N-step return
                n_step_state, n_step_action, n_step_reward_list, n_step_next_state, n_step_done = n_step_buffer[0] # First element is the N-step start
                cumulative_reward = 0
                discount_factor   = 1.0
                for i in range(len(n_step_buffer)): # Sum rewards in N-step buffer
                    cumulative_reward += discount_factor * n_step_buffer[i][2]
                    discount_factor   *= gamma
                n_step_reward     = cumulative_reward


                # sample нэмэхдээ temporal difference error-ийг тооцож нэмэх (using N-step transition)
                temporal_difference = float(td_error(optimizer.target, target_q_network, (
                        jnp.asarray([n_step_state]), # N-step start state
                        jnp.asarray([n_step_action]), # N-step action
                        jnp.asarray([n_step_reward]), # N-step cumulative reward
                        jnp.asarray([new_state]),     # N-step end state (current new_state)
                        jnp.asarray([discount_factor])# N-step discount factor
                    ))[0])
                per_memory.add(temporal_difference, (n_step_state, n_step_action, n_step_reward, new_state, int(done), discount_factor)) # Store N-step transition

            if len(per_memory) > batch_size:
                # Prioritized Experience Replay санах ойгоос batch үүсгээд DQN сүлжээг сургах (using N-step transitions)
                batch = per_memory.sample(batch_size)
                states, actions, n_step_rewards, next_states, dones, discount_factors = [], [], [], [], [], [] # Extract N-step data
                for i in range(batch_size):
                    states.append        (batch[i][1][0])
                    actions.append       (batch[i][1][1])
                    n_step_rewards.append(batch[i][1][2]) # N-step reward
                    next_states.append   (batch[i][1][3]) # N-step next state
                    dones.append         (batch[i][1][4])
                    discount_factors.append(batch[i][1][5]) # N-step discount factor


                optimizer, loss, new_td_errors = train_step( # Train step (using N-step batch)
                                            optimizer,
                                            target_q_network,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур
                                                # төхөөрөмжийн санах ойруу хуулах (N-step batch)
                                                jnp.asarray(states),
                                                jnp.asarray(actions),
                                                jnp.asarray(n_step_rewards),  # N-step rewards
                                                jnp.asarray(next_states),     # N-step next states
                                                jnp.asarray(dones),
                                                jnp.asarray(discount_factors) # N-step discount factors
                                            )
                                        )
                # batch-аас бий болсон temporal difference error-ийн дагуу санах ойг шинэчлэх (using N-step TD error)
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
            if len(n_step_buffer) == n_step_learning: # Clear N-step buffer after processing N-step transition
                n_step_buffer.popleft() # Remove the oldest transition
finally:
    env.close()