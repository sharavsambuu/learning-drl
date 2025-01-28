#
# sudo apt install python3.12-tk
#

import os
import random
import math
import matplotlib
matplotlib.use('TkAgg')  # try 'QtAgg' if TkAgg doesn't work
import matplotlib.pyplot as plt
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


debug_render  = True  
num_episodes  = 500
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 1000
epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01
gamma         = 0.99


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

class PERMemory:
    e = 0.01
    a = 0.6
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    def _get_priority(self, error):
        return (error + self.e) ** self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
    def sample(self, n):
        batch = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class DistributionalDQN(nn.Module):
    n_actions: int
    n_atoms  : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        outputs = []
        for _ in range(self.n_actions):
            x_atom = nn.Dense(features=self.n_atoms)(x)
            x_atom = nn.softmax(x_atom)
            outputs.append(x_atom)
        return jnp.stack(outputs, axis=0)



per_memory = PERMemory(memory_length)

env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n


v_min    = -10.0
v_max    = 10.0
n_atoms  = 51
dz       = float(v_max - v_min) / (n_atoms - 1)
z_holder = jnp.array([v_min + i * dz for i in range(n_atoms)])

nn_module   = DistributionalDQN(n_actions=n_actions, n_atoms=n_atoms)
dummy_input = jnp.zeros(state.shape)
params      = nn_module.init(jax.random.PRNGKey(0), dummy_input)['params']

nn_model_params        = params
target_nn_model_params = params

optimizer_def   = optax.adam(learning_rate)
optimizer_state = optimizer_def.init(nn_model_params)



@jax.jit
def inference(params, input_batch):
    return nn_module.apply({'params': params}, input_batch)

@jax.jit
def backpropagate(optimizer_state, model_params, target_model_params, states, actions, rewards, next_states, dones):
    def loss_fn(params):
        predicted_distributions = nn_module.apply({'params': params}, states)
        target_distributions = nn_module.apply({'params': target_model_params}, next_states)
        loss = jnp.mean(
            jax.vmap(categorical_loss_fn, in_axes=(1, 1, 0, 0, 0))(
                predicted_distributions,
                target_distributions,
                actions,
                rewards,
                dones
            )
        )
        return loss
    loss, gradients = jax.value_and_grad(loss_fn)(model_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, model_params)
    new_model_params = optax.apply_updates(model_params, updates)
    return new_optimizer_state, new_model_params, loss

@jax.jit
def categorical_loss_fn(predicted_distribution, target_distribution, action, reward, done):
    predicted_action_dist = predicted_distribution[action]
    projected_dist = projection_distribution(reward, done, target_distribution)
    return -jnp.sum(projected_dist * jnp.log(predicted_action_dist + 1e-6))

@jax.jit
def projection_distribution(reward, done, next_state_distribution):
    projected_distribution = jnp.zeros((n_atoms,))
    q_values = jnp.sum(next_state_distribution * jnp.expand_dims(z_holder, axis=0), axis=1)
    best_next_action = jnp.argmax(q_values)
    next_state_best_action_dist = next_state_distribution[best_next_action]

    def done_case(reward, gamma, z_holder, next_state_best_action_dist):
        proj_dist_done = jnp.zeros((n_atoms,))
        Tz = jnp.clip(reward, v_min, v_max)
        bj = (Tz - v_min) / dz
        lower_bound = jnp.int32(jnp.floor(bj))
        upper_bound = jnp.int32(jnp.ceil(bj))
        fraction_upper = bj - lower_bound
        fraction_lower = 1.0 - fraction_upper
        proj_dist_done = proj_dist_done.at[lower_bound].add(fraction_lower)
        proj_dist_done = proj_dist_done.at[upper_bound].add(fraction_upper)
        return proj_dist_done
    def not_done_case(reward, gamma, z_holder, next_state_best_action_dist):
        proj_dist_not_done = jnp.zeros((n_atoms,))
        for atom_index in range(n_atoms):
            Tz = jnp.clip(reward + gamma * z_holder[atom_index], v_min, v_max)
            bj = (Tz - v_min) / dz
            lower_bound = jnp.int32(jnp.floor(bj))
            upper_bound = jnp.int32(jnp.ceil(bj))
            fraction_upper = bj - lower_bound
            fraction_lower = 1.0 - fraction_upper

            proj_dist_not_done = proj_dist_not_done.at[lower_bound].add(fraction_lower * next_state_best_action_dist[atom_index])
            proj_dist_not_done = proj_dist_not_done.at[upper_bound].add(fraction_upper * next_state_best_action_dist[atom_index])
        return proj_dist_not_done
    projected_distribution = jax.lax.cond(
        done,
        lambda reward, gamma, z_holder, next_state_best_action_dist: done_case(reward, gamma, z_holder, next_state_best_action_dist),
        lambda reward, gamma, z_holder, next_state_best_action_dist: not_done_case(reward, gamma, z_holder, next_state_best_action_dist),
        reward, gamma, z_holder, next_state_best_action_dist
    )
    return projected_distribution


if debug_render:
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

plt.ion()  # Ensure matplotlib is in interactive mode

global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)

        while True:
            global_steps += 1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                outputs = inference(nn_model_params, jnp.array([state]))
                try: # try-except block to catch plotting errors
                    if debug_render: # Redundant if statement, but keep for clarity
                        plt.clf()
                        # Plotting distribution for each action
                        for action_index in range(n_actions):
                            plt.bar(z_holder, outputs[0][action_index], alpha=0.5, label=f'Action {action_index}')
                        plt.xlabel("Atom Value (Z)")
                        plt.ylabel("Probability")
                        plt.title("Categorical Distribution of Atoms")
                        plt.legend(loc='upper left')
                        fig.canvas.draw()
                        plt.show(block=False) # Explicitly show the plot, non-blocking
                        plt.pause(0.001) # tiny pause to allow plot to update
                except Exception as e:
                    print(f"Error during plotting: {e}") # Debug print 3: Catch and print plotting errors

                q = jnp.sum(outputs[0] * z_holder, axis=1)
                action = int(jnp.argmax(q))

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            target_distributions_next_state = inference(target_nn_model_params, jnp.array(new_state))
            predicted_distributions_current_state = inference(nn_model_params, jnp.array([state]))

            td_error = categorical_loss_fn(
                predicted_distributions_current_state[0],
                target_distributions_next_state,
                action,
                reward,
                done,
            )

            per_memory.add(td_error, (state, action, reward, new_state, int(done)))

            batch = per_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            idxs = []
            for i in range(batch_size):
                idxs       .append(batch[i][0])
                states     .append(batch[i][1][0])
                actions    .append(batch[i][1][1])
                rewards    .append(batch[i][1][2])
                next_states.append(batch[i][1][3])
                dones      .append(batch[i][1][4])

            optimizer_state, nn_model_params, loss = backpropagate(
                optimizer_state, nn_model_params, target_nn_model_params,
                jnp.array(states     ),
                jnp.array(actions    ),
                jnp.array(rewards    ),
                jnp.array(next_states),
                jnp.array(dones      )
            )

            predicted_distributions_batch = inference(nn_model_params, jnp.array(states))
            new_priorities = np.zeros(batch_size)

            for i in range(batch_size):
                target_distributions_batch = inference(target_nn_model_params, jnp.array([next_states[i]]))
                td_error_sample = categorical_loss_fn(
                    predicted_distributions_batch[i],
                    target_distributions_batch[0],
                    actions[i],
                    rewards[i],
                    dones[i],
                )
                new_priorities[i] = td_error_sample

            for i in range(batch_size):
                per_memory.update(idxs[i], new_priorities[i])

            episode_rewards.append(reward)
            state = new_state

            if global_steps % sync_steps == 0:
                target_nn_model_params = nn_model_params

            if debug_render:
                env.render()

            if done:
                print("{} episode, reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()