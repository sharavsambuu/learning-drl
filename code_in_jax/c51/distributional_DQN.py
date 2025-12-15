#
# Distributional DQN (C51) + PER + Target Network + Live Atom Plot (CartPole-v1)
#
# Implemented:
#   - C51 distributional Q-learning (categorical atoms on fixed support [v_min, v_max])
#   - Prioritized Experience Replay (PER) using a SumTree
#   - Target network sync every sync_steps
#   - Live matplotlib plot of the atom distributions per action
#
# Improved:
#   1) Network output shape is (batch, n_actions, n_atoms)    (axis=1 stack)
#   2) Inference always gets a batch dimension               
#   3) PER tracks actual number of inserted samples           (n_entries)
#   4) C51 projection is vectorized + JAX-friendly            (no Python loops inside jit)
#   5) Priority update is batch-computed (fast + consistent) 
#
# Notes:
#   - CartPole rewards are {1 per step}, so v_min/v_max can be tighter, but keep as-is for now.
#   - PER priority uses categorical loss (reasonable proxy).
#   - sudo apt install python3.13-tk
#
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
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data      = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
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
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx               = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx     = self._retrieve(0, s)
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
        batch   = []
        total_p = self.tree.total()
        if total_p <= 0:
            return batch

        segment = total_p / n
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

    def size(self):
        return self.tree.n_entries


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

        # (batch, n_actions, n_atoms)
        return jnp.stack(outputs, axis=1)


per_memory = PERMemory(memory_length)

env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n


v_min                  = 0.0
v_max                  = 100.0
n_atoms                = 51
dz                     = float(v_max - v_min) / (n_atoms - 1)
z_holder               = jnp.array([v_min + i * dz for i in range(n_atoms)], dtype=jnp.float32)


nn_module              = DistributionalDQN(n_actions=n_actions, n_atoms=n_atoms)
dummy_input            = jnp.zeros(state.shape, dtype=jnp.float32)
params                 = nn_module.init(jax.random.PRNGKey(0), dummy_input)['params']

nn_model_params        = params
target_nn_model_params = params

optimizer_def          = optax.adam(learning_rate)
optimizer_state        = optimizer_def.init(nn_model_params)


@jax.jit
def inference(params, input_batch):
    return nn_module.apply({'params': params}, input_batch)

@jax.jit
def projection_distribution(reward, done, next_state_distribution):
    # next_state_distribution: (n_actions, n_atoms)

    # pick best action by expected value
    q_values         = jnp.sum(next_state_distribution * z_holder[None, :], axis=1)
    best_next_action = jnp.argmax(q_values)
    p_next           = next_state_distribution[best_next_action]  # (n_atoms,)

    def done_case(reward, p_next):
        m  = jnp.zeros((n_atoms,), dtype=jnp.float32)
        Tz = jnp.clip(reward, v_min, v_max)
        bj = (Tz - v_min) / dz
        l  = jnp.int32(jnp.floor(bj))
        u  = jnp.int32(jnp.ceil(bj))

        # if l == u, put all mass there
        same = (l == u).astype(jnp.float32)
        m    = m.at[l].add(same * 1.0)

        # otherwise interpolate
        frac_u = bj - l
        frac_l = 1.0 - frac_u
        m      = m.at[l].add((1.0 - same) * frac_l)
        m      = m.at[u].add((1.0 - same) * frac_u)
        return m

    def not_done_case(reward, p_next):
        m  = jnp.zeros((n_atoms,), dtype=jnp.float32)

        Tz = jnp.clip(reward + gamma * z_holder, v_min, v_max)   # (n_atoms,)
        bj = (Tz - v_min) / dz                                   # (n_atoms,)
        l  = jnp.int32(jnp.floor(bj))                            # (n_atoms,)
        u  = jnp.int32(jnp.ceil(bj))                             # (n_atoms,)

        same = (l == u).astype(jnp.float32)

        frac_u = bj - l
        frac_l = 1.0 - frac_u

        # if l == u, add full mass p_next
        m = m.at[l].add(p_next * same)

        # otherwise distribute
        m = m.at[l].add(p_next * (1.0 - same) * frac_l)
        m = m.at[u].add(p_next * (1.0 - same) * frac_u)
        return m

    return jax.lax.cond(
        done                                                ,
        lambda reward, p_next: done_case(reward, p_next)    ,
        lambda reward, p_next: not_done_case(reward, p_next),
        reward, p_next
    )

@jax.jit
def categorical_loss_fn(predicted_distribution, target_distribution, action, reward, done):
    # predicted_distribution: (n_actions, n_atoms)
    # target_distribution   : (n_actions, n_atoms)
    predicted_action_dist = predicted_distribution[action]                              # (n_atoms,)
    projected_dist        = projection_distribution(reward, done, target_distribution)  # (n_atoms,)
    return -jnp.sum(projected_dist * jnp.log(predicted_action_dist + 1e-6))

@jax.jit
def backpropagate(optimizer_state, model_params, target_model_params, states, actions, rewards, next_states, dones):
    def loss_fn(params):
        predicted_distributions = nn_module.apply({'params': params             }, states     )  # (B, A, Z)
        target_distributions    = nn_module.apply({'params': target_model_params}, next_states)  # (B, A, Z)

        losses = jax.vmap(categorical_loss_fn, in_axes=(0, 0, 0, 0, 0))(
            predicted_distributions,
            target_distributions   ,
            actions                ,
            rewards                ,
            dones
        )
        return jnp.mean(losses)

    loss, gradients              = jax.value_and_grad(loss_fn)(model_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, model_params)
    new_model_params             = optax.apply_updates(model_params, updates)
    return new_optimizer_state, new_model_params, loss


if debug_render:
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    ax_action0 = axes[0]
    ax_action1 = axes[1]
    ax_overlap = axes[2]

    color_action0 = "tab:blue"
    color_action1 = "tab:orange"

    for ax in axes:
        ax.set_xlim(float(v_min), float(v_max))
        ax.set_ylim(0.0, 1.0)  # distributions cap
        ax.set_xlabel("Atom Value (Z)")
        ax.set_ylabel("Probability")

    ax_action0.set_title("Action 0")
    ax_action1.set_title("Action 1")
    ax_overlap.set_title("Overlap")

    fig.tight_layout()
    fig.show()
    fig.canvas.draw()


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
                outputs = inference(nn_model_params, jnp.asarray([state], dtype=jnp.float32))  # (1, A, Z)

                try:
                    if debug_render:
                        dist0 = np.array(outputs[0][0])  # Action 0 dist (n_atoms,)
                        dist1 = np.array(outputs[0][1])  # Action 1 dist (n_atoms,)

                        ax_action0.cla()
                        ax_action1.cla()
                        ax_overlap.cla()

                        # Re-apply consistent scales/labels (cla() wipes them)
                        for ax in (ax_action0, ax_action1, ax_overlap):
                            ax.set_xlim(float(v_min), float(v_max))
                            ax.set_ylim(0.0, 1.0)
                            ax.set_xlabel("Atom Value (Z)")
                            ax.set_ylabel("Probability")

                        ax_action0.set_title("Action 0")
                        ax_action1.set_title("Action 1")
                        ax_overlap.set_title("Overlap")

                        # Plot action-specific
                        ax_action0.bar(np.array(z_holder), dist0, alpha=0.85, color=color_action0)
                        ax_action1.bar(np.array(z_holder), dist1, alpha=0.85, color=color_action1)

                        # Plot overlap (same colors)
                        ax_overlap.bar(np.array(z_holder), dist0, alpha=0.55, color=color_action0, label="Action 0")
                        ax_overlap.bar(np.array(z_holder), dist1, alpha=0.55, color=color_action1, label="Action 1")
                        ax_overlap.legend(loc="upper left")

                        fig.canvas.draw()
                        plt.show(block=False)
                        plt.pause(0.0001)
                except Exception as e:
                    print(f"Error during plotting: {e}")

                q = jnp.sum(outputs[0] * z_holder[None, :], axis=1)  # (A,)
                action = int(jnp.argmax(q))

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            # PER: add transition with priority
            predicted_current = inference(nn_model_params       , jnp.asarray([state    ], dtype=jnp.float32))[0]  # (A, Z)
            target_next       = inference(target_nn_model_params, jnp.asarray([new_state], dtype=jnp.float32))[0]  # (A, Z)

            td_error = categorical_loss_fn(
                predicted_current                        ,
                target_next                              ,
                jnp.asarray(action   , dtype=jnp.int32  ),
                jnp.asarray(reward   , dtype=jnp.float32),
                jnp.asarray(int(done), dtype=jnp.int32  ),
            )
            per_memory.add(float(td_error), (state, action, reward, new_state, int(done)))

            # Training
            if per_memory.size() >= batch_size:
                batch = per_memory.sample(batch_size)

                states, actions, rewards, next_states, dones = [], [], [], [], []
                idxs = []
                for i in range(batch_size):
                    idxs       .append(batch[i][0]   )
                    states     .append(batch[i][1][0])
                    actions    .append(batch[i][1][1])
                    rewards    .append(batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones      .append(batch[i][1][4])

                optimizer_state, nn_model_params, loss = backpropagate(
                    optimizer_state, nn_model_params, target_nn_model_params,
                    jnp.asarray(states     , dtype=jnp.float32),
                    jnp.asarray(actions    , dtype=jnp.int32  ),
                    jnp.asarray(rewards    , dtype=jnp.float32),
                    jnp.asarray(next_states, dtype=jnp.float32),
                    jnp.asarray(dones      , dtype=jnp.int32  )
                )

                # Update priorities (batch-wise)
                predicted_distributions_batch = inference(nn_model_params       , jnp.asarray(states     , dtype=jnp.float32))  # (B, A, Z)
                target_distributions_batch    = inference(target_nn_model_params, jnp.asarray(next_states, dtype=jnp.float32))  # (B, A, Z)

                td_errors_batch = jax.vmap(categorical_loss_fn, in_axes=(0, 0, 0, 0, 0))(
                    predicted_distributions_batch          ,
                    target_distributions_batch             ,
                    jnp.asarray(actions, dtype=jnp.int32  ),
                    jnp.asarray(rewards, dtype=jnp.float32),
                    jnp.asarray(dones  , dtype=jnp.int32  ),
                )
                td_errors_batch = np.array(td_errors_batch)

                for i in range(batch_size):
                    per_memory.update(idxs[i], float(td_errors_batch[i]))

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
