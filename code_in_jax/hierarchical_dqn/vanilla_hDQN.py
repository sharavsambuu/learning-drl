#
# hDQN (Goal-Conditioned Controller + Meta-Controller) + PER + Target Networks
#
#   Meta-controller picks a GOAL id every meta_controller_interval steps (or earlier if goal reached).
#   Controller picks primitive action conditioned on (state, goal_id) using intrinsic reward.
#   Meta learns on k-step accumulated extrinsic reward with gamma**k bootstrap.
#

import os
import random
import math
import gymnasium as gym
from collections import deque

import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


debug_render             = True
debug                    = False
num_episodes             = 500

batch_size               = 128
learning_rate            = 0.001
sync_steps               = 250
memory_length            = 4000

meta_controller_interval = 10

epsilon                  = 1.0
epsilon_decay            = 0.0005
epsilon_max              = 1.0
epsilon_min              = 0.01

gamma                    = 0.99

goal_positions           = jnp.asarray([-0.5, 0.0, 0.5], dtype=jnp.float32)
goal_threshold           = 0.08


class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2*capacity - 1, dtype=np.float32)

        # unfilled slots must be None, not 0 (int), otherwise d[0] crashes
        self.data      = np.empty(capacity, dtype=object)
        self.data[:]   = None

        self.n_entries = 0
        self.write     = 0

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
        return float(self.tree[0])

    def add(self, p, data):
        idx                   = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write           += 1
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

        # If retrieval lands on an unfilled leaf, return None so sampler can retry.
        if dataIdx >= self.n_entries:
            return (idx, self.tree[idx], None)

        data = self.data[dataIdx]
        if data is None:
            return (idx, self.tree[idx], None)

        return (idx, self.tree[idx], data)


class PERMemory:
    e = 0.01
    a = 0.7
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return float((error + self.e) ** self.a)

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch     = []
        n_entries = self.size()
        if n_entries <= 0:
            return batch

        n = min(n, n_entries)

        total_p = self.tree.total()
        if total_p <= 0:
            return batch

        segment = total_p / n
        eps     = 1e-6

        for i in range(n):
            a = segment * i
            b = segment * (i+1)

            # Retry within segment to avoid boundary hits that can land on empty/zero-priority subtrees.
            for _ in range(10):
                width = b - a
                if width <= eps:
                    s = a + 0.5 * width
                else:
                    s = a + eps + (width - eps) * random.random()  # (a, b)

                (idx, p, data) = self.tree.get(s)
                if data is not None:
                    batch.append((idx, data))
                    break

        return batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class MetaControllerNetwork(nn.Module):
    n_goals: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_goals)(x)
        return x

class ControllerNetwork(nn.Module):
    n_actions: int
    n_goals  : int
    @nn.compact
    def __call__(self, x, goal_id):
        goal_one_hot = jax.nn.one_hot(goal_id, self.n_goals)
        combined     = jnp.concatenate([x, goal_one_hot], axis=-1)
        x = nn.Dense(features=64)(combined)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x


env         = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)

n_actions   = env.action_space.n
n_goals     = int(goal_positions.shape[0])

meta_module = MetaControllerNetwork(n_goals=n_goals)
ctrl_module = ControllerNetwork(n_actions=n_actions, n_goals=n_goals)

key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)

meta_params = meta_module.init(k1, jnp.asarray(state, dtype=jnp.float32))['params']
tgt_meta_params = meta_params

ctrl_params = ctrl_module.init(
    k2,
    jnp.asarray(state, dtype=jnp.float32),
    jnp.asarray(0, dtype=jnp.int32)
)['params']
tgt_ctrl_params = ctrl_params

meta_optimizer = optax.adam(learning_rate)
ctrl_optimizer = optax.adam(learning_rate)

meta_opt_state = meta_optimizer.init(meta_params)
ctrl_opt_state = ctrl_optimizer.init(ctrl_params)

meta_memory = PERMemory(memory_length)
ctrl_memory = PERMemory(memory_length)


@jax.jit
def meta_policy(params, x):
    q = meta_module.apply({'params': params}, x)
    a = jnp.argmax(q)
    return a, q

@jax.jit
def ctrl_policy(params, x, goal_id):
    q = ctrl_module.apply({'params': params}, x, goal_id)
    a = jnp.argmax(q)
    return a, q


def td_error_dqn(q_value_vec, target_q_value_vec, action, reward, done):
    one_hot_actions = jax.nn.one_hot(action, q_value_vec.shape[-1])
    q_value         = jnp.sum(one_hot_actions * q_value_vec, axis=-1)
    td_target       = reward + gamma * jnp.max(target_q_value_vec, axis=-1) * (1.0 - done)
    td_error        = td_target - q_value
    return jnp.abs(td_error)

td_error_dqn_vmap = jax.vmap(td_error_dqn, in_axes=(0, 0, 0, 0, 0), out_axes=0)

def loss_dqn(q_value_vec, target_q_value_vec, action, reward, done):
    one_hot_actions = jax.nn.one_hot(action, q_value_vec.shape[-1])
    q_value         = jnp.sum(one_hot_actions * q_value_vec, axis=-1)
    td_target       = reward + gamma * jnp.max(target_q_value_vec, axis=-1) * (1.0 - done)
    td_error        = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

loss_dqn_vmap = jax.vmap(loss_dqn, in_axes=(0, 0, 0, 0, 0), out_axes=0)


@jax.jit
def train_step_meta(meta_params, tgt_meta_params, meta_opt_state, batch):
    def loss_fn(params):
        q      = meta_module.apply({'params': params}, batch[0])
        q_next = meta_module.apply({'params': tgt_meta_params}, batch[3])

        a      = batch[1]
        r_sum  = batch[2]
        done   = batch[4]
        k_step = batch[5]

        one_hot_a = jax.nn.one_hot(a, n_goals)
        q_sa      = jnp.sum(q * one_hot_a, axis=-1)

        discount  = jnp.power(gamma, k_step)
        target    = r_sum + (1.0 - done) * discount * jnp.max(q_next, axis=-1)

        td_error  = jax.lax.stop_gradient(target) - q_sa
        return jnp.mean(jnp.square(td_error))

    loss, grads = jax.value_and_grad(loss_fn)(meta_params)
    updates, meta_opt_state = meta_optimizer.update(grads, meta_opt_state, meta_params)
    meta_params = optax.apply_updates(meta_params, updates)

    q      = meta_module.apply({'params': meta_params}, batch[0])
    q_next = meta_module.apply({'params': tgt_meta_params}, batch[3])
    one_hot_a = jax.nn.one_hot(batch[1], n_goals)
    q_sa      = jnp.sum(q * one_hot_a, axis=-1)
    discount  = jnp.power(gamma, batch[5])
    target    = batch[2] + (1.0 - batch[4]) * discount * jnp.max(q_next, axis=-1)
    td_err    = jnp.abs(target - q_sa)

    return meta_params, meta_opt_state, loss, td_err


@jax.jit
def train_step_ctrl(ctrl_params, tgt_ctrl_params, ctrl_opt_state, batch):
    def loss_fn(params):
        q = ctrl_module.apply({'params': params}, batch[0], batch[1])
        q_next = ctrl_module.apply({'params': tgt_ctrl_params}, batch[3], batch[1])
        losses = loss_dqn_vmap(q, q_next, batch[2], batch[4], batch[5])
        return jnp.mean(losses)

    loss, grads = jax.value_and_grad(loss_fn)(ctrl_params)
    updates, ctrl_opt_state = ctrl_optimizer.update(grads, ctrl_opt_state, ctrl_params)
    ctrl_params = optax.apply_updates(ctrl_params, updates)

    q = ctrl_module.apply({'params': ctrl_params}, batch[0], batch[1])
    q_next = ctrl_module.apply({'params': tgt_ctrl_params}, batch[3], batch[1])
    td_err = td_error_dqn_vmap(q, q_next, batch[2], batch[4], batch[5])

    return ctrl_params, ctrl_opt_state, loss, td_err


def goal_reached(state_np, goal_id):
    x = float(state_np[0])
    gx = float(goal_positions[int(goal_id)])
    return abs(x - gx) <= float(goal_threshold)

def intrinsic_reward(state_np, goal_id):
    return 1.0 if goal_reached(state_np, goal_id) else 0.0


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)

        meta_goal_start_state = state.copy()
        meta_goal_id          = 0
        meta_steps_in_goal    = 0
        meta_extrinsic_sum    = 0.0

        while True:
            global_steps += 1

            need_new_goal = False
            if meta_steps_in_goal == 0:
                need_new_goal = True
            if meta_steps_in_goal >= meta_controller_interval:
                need_new_goal = True
            if goal_reached(state, meta_goal_id):
                need_new_goal = True

            if need_new_goal:
                if meta_steps_in_goal > 0:
                    meta_memory.add(
                        1e3,
                        (
                            meta_goal_start_state,
                            int(meta_goal_id),
                            float(meta_extrinsic_sum),
                            state.copy(),
                            0.0,
                            float(meta_steps_in_goal),
                        )
                    )

                meta_goal_start_state = state.copy()
                meta_steps_in_goal    = 0
                meta_extrinsic_sum    = 0.0

                if np.random.rand() <= epsilon:
                    meta_goal_id = np.random.randint(0, n_goals)
                else:
                    g, _ = meta_policy(meta_params, jnp.asarray(state, dtype=jnp.float32))
                    meta_goal_id = int(g)

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                a, _ = ctrl_policy(
                    ctrl_params,
                    jnp.asarray(state, dtype=jnp.float32),
                    jnp.asarray(meta_goal_id, dtype=jnp.int32)
                )
                action = int(a)

            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32)

            ir = intrinsic_reward(new_state, meta_goal_id)

            ctrl_memory.add(
                1e3,
                (
                    state.copy(),
                    int(meta_goal_id),
                    int(action),
                    float(ir),
                    new_state.copy(),
                    float(done or goal_reached(new_state, meta_goal_id)),
                )
            )

            meta_steps_in_goal += 1
            meta_extrinsic_sum += float(reward)

            episode_rewards.append(float(reward))
            state = new_state

            if ctrl_memory.size() >= batch_size:
                batch = ctrl_memory.sample(batch_size)
                if len(batch) < batch_size:
                    continue

                idxs, data = zip(*batch)

                states, goals, actions, i_rewards, next_states, g_dones = [], [], [], [], [], []
                for d in data:
                    states     .append(d[0])
                    goals      .append(d[1])
                    actions    .append(d[2])
                    i_rewards  .append(d[3])
                    next_states.append(d[4])
                    g_dones    .append(d[5])

                ctrl_params, ctrl_opt_state, ctrl_loss, ctrl_td = train_step_ctrl(
                    ctrl_params, tgt_ctrl_params, ctrl_opt_state,
                    (
                        jnp.asarray(states     , dtype=jnp.float32),
                        jnp.asarray(goals      , dtype=jnp.int32  ),
                        jnp.asarray(actions    , dtype=jnp.int32  ),
                        jnp.asarray(next_states, dtype=jnp.float32),
                        jnp.asarray(i_rewards  , dtype=jnp.float32),
                        jnp.asarray(g_dones    , dtype=jnp.float32),
                    )
                )

                ctrl_td_np = np.array(ctrl_td)
                for i in range(batch_size):
                    ctrl_memory.update(idxs[i], float(ctrl_td_np[i]))

            if meta_memory.size() >= batch_size:
                batch = meta_memory.sample(batch_size)
                if len(batch) < batch_size:
                    continue

                idxs, data = zip(*batch)

                s0, g, rsum, sk, dones, ksteps = [], [], [], [], [], []
                for d in data:
                    s0    .append(d[0])
                    g     .append(d[1])
                    rsum  .append(d[2])
                    sk    .append(d[3])
                    dones .append(d[4])
                    ksteps.append(d[5])

                meta_params, meta_opt_state, meta_loss, meta_td = train_step_meta(
                    meta_params, tgt_meta_params, meta_opt_state,
                    (
                        jnp.asarray(s0    , dtype=jnp.float32),
                        jnp.asarray(g     , dtype=jnp.int32  ),
                        jnp.asarray(rsum  , dtype=jnp.float32),
                        jnp.asarray(sk    , dtype=jnp.float32),
                        jnp.asarray(dones , dtype=jnp.float32),
                        jnp.asarray(ksteps, dtype=jnp.float32),
                    )
                )

                meta_td_np = np.array(meta_td)
                for i in range(batch_size):
                    meta_memory.update(idxs[i], float(meta_td_np[i]))

            if global_steps % sync_steps == 0:
                tgt_meta_params = meta_params
                tgt_ctrl_params = ctrl_params

            if debug_render:
                env.render()

            if done:
                meta_memory.add(
                    1e3,
                    (
                        meta_goal_start_state,
                        int(meta_goal_id),
                        float(meta_extrinsic_sum),
                        state.copy(),
                        1.0,
                        float(meta_steps_in_goal),
                    )
                )
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break
finally:
    env.close()
