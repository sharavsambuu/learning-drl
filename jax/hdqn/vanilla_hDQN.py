#
# h-DQN aka Hierarchical Deep Q-Network
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

debug_render             = False 
num_episodes             = 500
batch_size               = 64
learning_rate            = 0.001
sync_steps               = 100
memory_length            = 4000
meta_controller_interval = 10     # Meta-controller makes decision every N steps
epsilon                  = 1.0
epsilon_decay            = 0.001
epsilon_max              = 1.0
epsilon_min              = 0.01
gamma                    = 0.99   # discount factor


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


class MetaControllerNetwork(flax.nn.Module): # Meta-Controller Network
    def apply(self, x, n_meta_actions): # Output Q-values for meta-actions
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_meta_actions)
        return output_layer

class ControllerNetwork(flax.nn.Module): # Controller Network
    def apply(self, x, meta_action, n_actions): # Takes state and meta-action as input
        combined_input   = jnp.concatenate([x, jnp.expand_dims(meta_action, axis=-1)], -1) # Conditioned on meta-action
        dense_layer_1      = flax.nn.Dense(combined_input, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_actions) # Output Q-values for primitive actions
        return output_layer


env   = gym.make('CartPole-v1')
state = env.reset()
n_actions        = env.action_space.n
n_meta_actions   = n_actions # meta-actions are same as primitive actions


meta_controller_module = MetaControllerNetwork.partial(n_meta_actions=n_meta_actions)
_, meta_controller_params = meta_controller_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
meta_controller_model = flax.nn.Model(meta_controller_module, meta_controller_params)
target_meta_controller_model = flax.nn.Model(meta_controller_module, meta_controller_params) # Target Meta-Controller

controller_module = ControllerNetwork.partial(n_actions=n_actions)
_, controller_params = controller_module.init_by_shape(jax.random.PRNGKey(0), [state.shape], [()]) # Controller takes state and meta-action
controller_model = flax.nn.Model(controller_module, controller_params)
target_controller_model = flax.nn.Model(controller_module, controller_params) # Target Controller

meta_controller_optimizer = flax.optim.Adam(learning_rate).create(meta_controller_model)
controller_optimizer = flax.optim.Adam(learning_rate).create(controller_model)

per_memory       = PERMemory(memory_length)


@jax.jit
def meta_controller_policy(model, x): # Policy for Meta-Controller
    predicted_q_values = model(x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values

@jax.jit
def controller_policy(model, x, meta_action): # Policy for Controller (conditioned on meta-action)
    predicted_q_values = model(x, meta_action)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


@jax.jit
def calculate_td_error(q_value_vec, target_q_value_vec, action, reward): # TD Error calculation 
    td_target = reward + gamma*jnp.max(target_q_value_vec)
    td_error  = td_target - q_value_vec[action]
    return jnp.abs(td_error)

@jax.jit
def td_error_meta_controller(meta_controller_model, target_meta_controller_model, batch): # TD Error for Meta-Controller
    # Batch is same as DQN batch
    states, meta_actions, rewards, next_states, dones = batch
    predicted_q_values      = meta_controller_model(states)
    target_q_values         = target_meta_controller_model(next_states)
    return calculate_td_error(predicted_q_values, target_q_values, meta_actions, rewards)

@jax.jit
def td_error_controller(controller_model, target_controller_model, batch, meta_action): # TD Error for Controller (conditioned on meta-action)
    # Batch is same as DQN batch
    states, actions, rewards, next_states, dones = batch
    predicted_q_values      = controller_model(states, meta_action)
    target_q_values         = target_controller_model(next_states, meta_action) # Still condition target network on meta-action
    return calculate_td_error(predicted_q_values, target_q_values, actions, rewards)


@jax.vmap # vmap for batch loss calculation 
def q_learning_loss(q_value_vec, target_q_value_vec, action, reward, done):
    td_target = reward + gamma*jnp.max(target_q_value_vec)*(1.-done)
    td_error  = jax.lax.stop_gradient(td_target) - q_value_vec[action]
    return jnp.square(td_error)

@jax.jit
def train_step_meta_controller(optimizer, target_model, batch): # Train step for Meta-Controller 
    # batch[0] - states
    # batch[1] - meta_actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        predicted_q_values      = model(batch[0])
        target_q_values         = target_model(batch[3])
        return jnp.mean(
                q_learning_loss(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error_meta_controller(optimizer.target, target_model, batch) # Use meta-controller TD error

@jax.jit
def train_step_controller(optimizer, target_model, batch, meta_action): # Train step for Controller (conditioned on meta-action)
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        predicted_q_values      = model(batch[0], meta_action) # Conditioned on meta-action
        target_q_values         = target_model(batch[3], meta_action) # Still condition target on meta-action
        return jnp.mean(
                q_learning_loss(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error_controller(optimizer.target, target_model, batch, meta_action) # Use controller TD error


global_steps = 0
meta_action = None # Initialize meta-action outside episode loop
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        meta_controller_steps = 0 # Counter for meta-controller interval
        while True:
            global_steps += 1
            meta_controller_steps += 1

            if meta_controller_steps % meta_controller_interval == 1: # Meta-controller decides every interval steps
                if np.random.rand() <= epsilon:
                    meta_action = env.action_space.sample() # Meta-action exploration
                else:
                    meta_action, _ = meta_controller_policy(meta_controller_optimizer.target, jnp.asarray([state])) # Meta-action from Meta-Controller Network
                meta_controller_steps = 1 # Reset counter after meta-action selection


            if np.random.rand() <= epsilon:
                action = env.action_space.sample() # Controller action exploration
            else:
                action, q_values = controller_policy(controller_optimizer.target, jnp.asarray([state]), jnp.asarray(meta_action)) # Controller action conditioned on meta-action


            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            new_state, reward, done, _ = env.step(int(action))

            # Replay buffer - Store experience with meta-action
            temporal_difference = 0.0 # Initialize temporal difference
            per_memory.add(temporal_difference, (state, meta_action, reward, new_state, int(done), action)) # Store meta_action in memory

            # Train every step (you can adjust training frequency)
            if len(per_memory) > batch_size:
                batch = per_memory.sample(batch_size)
                states, meta_actions_batch, rewards, next_states, dones, actions = [], [], [], [], [], [] # Extract meta_actions from batch
                for i in range(batch_size):
                    states.append     (batch[i][1][0])
                    meta_actions_batch.append(batch[i][1][1]) # meta_action from memory
                    rewards.append    (batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones.append      (batch[i][1][4])
                    actions.append    (batch[i][1][5]) # primitive action from memory


                # Train Meta-Controller Network
                meta_controller_optimizer, meta_controller_loss, _ = train_step_meta_controller(
                    meta_controller_optimizer, target_meta_controller_model,
                    (jnp.asarray(states), jnp.asarray(meta_actions_batch), jnp.asarray(rewards), jnp.asarray(next_states), jnp.asarray(dones))
                )
                # Train Controller Network - Conditioned on a *fixed* meta_action for the batch
                controller_optimizer, controller_loss, new_td_errors = train_step_controller(
                    controller_optimizer, target_controller_model,
                    (jnp.asarray(states), jnp.asarray(actions), jnp.asarray(rewards), jnp.asarray(next_states), jnp.asarray(dones)),
                    jnp.asarray(meta_action) # Condition Controller training on the *current* meta_action
                )
                # Update PER memory with controller TD errors (you can choose to use either controller or meta-controller TD error)
                new_td_errors = np.array(new_td_errors)
                for i in range(batch_size):
                    idx = batch[i][0]
                    per_memory.update(idx, new_td_errors[i])


            episode_rewards.append(reward)
            state = new_state

            if global_steps%sync_steps==0: # Target network updates (both Meta-Controller and Controller)
                target_meta_controller_model = target_meta_controller_model.replace(params=meta_controller_optimizer.target.params)
                target_controller_model = target_controller_model.replace(params=controller_optimizer.target.params)


            if debug_render:
                env.render()

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break
finally:
    env.close()