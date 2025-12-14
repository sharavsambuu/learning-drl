import random
import math
import jax
import optax
from   collections import deque
from   jax         import numpy as jnp
import numpy       as np
import gymnasium   as gym
import flax.linen  as nn


debug_render  = True   
debug         = False

num_episodes  = 500
batch_size    = 64
learning_rate = 1e-3
sync_steps    = 100
memory_length = 4000

n_steps       = 3
gamma         = 0.99

epsilon_max   = 1.0
epsilon_min   = 0.01
epsilon_decay = 0.001


class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data     = np.zeros(capacity, dtype=object)
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
        return self._retrieve(right, s - self.tree[left])
    def total(self):
        return float(self.tree[0])
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
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

class PERMemory:
    e = 0.01
    a = 0.6
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.size = 0  # track filled count
    def _get_priority(self, error):
        return float((error + self.e) ** self.a)
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.size = min(self.size + 1, self.capacity)
    def sample(self, n):
        batch = []
        total = self.tree.total()
        if total <= 0:
            return batch
        segment = total / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append((idx, data))
        return batch
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class DeepQNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x


env           = gym.make("CartPole-v1", render_mode="human" if debug_render else None)
state, info   = env.reset()
state         = np.array(state, dtype=np.float32)
n_actions     = env.action_space.n

dqn_module    = DeepQNetwork(n_actions=n_actions)
params        = dqn_module.init(jax.random.PRNGKey(0), jnp.zeros(state.shape))["params"]
q_params      = params
tgt_params    = params

optimizer     = optax.adam(learning_rate)
opt_state     = optimizer.init(q_params)

per_memory    = PERMemory(memory_length)
n_step_buffer = deque(maxlen=n_steps)  # (s, a, r, ns, done)


@jax.jit
def policy(params, x):
    q = dqn_module.apply({"params": params}, x)
    a = jnp.argmax(q)
    return a, q


def make_n_step_sample(buffer, gamma, n_steps):
    """
    Build one sample from the LEFT of buffer.
    Returns (s0, a0, R, s_k, done_k, gamma_pow), gamma_pow = gamma^k.
    """
    s0, a0, _, _, _ = buffer[0]

    r      = 0.0
    done_k = 0.0
    s_k    = buffer[-1][3]
    k      = 0

    for i, (s, a, r, ns, d) in enumerate(buffer):
        r     += (gamma ** i) * float(r)
        k      = i + 1
        s_k    = ns
        done_k = float(d)
        if d or k >= n_steps:
            break

    gamma_pow = float(gamma ** k)
    return s0, a0, float(r), s_k, float(done_k), float(gamma_pow)


def td_error_single(q_params, tgt_params, sample):
    """Compute abs TD error for one n-step sample."""
    s0, a0, r, s_k, done_k, gamma_pow = sample
    batch = (
        jnp.asarray([s0       ]                   ),
        jnp.asarray([a0       ], dtype=jnp.int32  ),
        jnp.asarray([r        ], dtype=jnp.float32),
        jnp.asarray([s_k      ]                   ),
        jnp.asarray([done_k   ], dtype=jnp.float32),
        jnp.asarray([gamma_pow], dtype=jnp.float32),
    )
    return float(td_error_batch(q_params, tgt_params, batch)[0])


def calculate_n_step_td_error(q_vec, tgt_vec, action, r, done, gamma_pow):
    q_sa      = jnp.sum(jax.nn.one_hot(action, n_actions) * q_vec)
    bootstrap = gamma_pow * jnp.max(tgt_vec) * (1.0 - done)
    target    = r + bootstrap
    return jnp.abs(target - q_sa)

calculate_n_step_td_error_vmap = jax.vmap(
    calculate_n_step_td_error, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0
)


def n_step_loss(q_vec, tgt_vec, action, r, done, gamma_pow):
    q_sa      = jnp.sum(jax.nn.one_hot(action, n_actions) * q_vec)
    bootstrap = gamma_pow * jnp.max(tgt_vec) * (1.0 - done)
    target    = r + bootstrap
    td        = jax.lax.stop_gradient(target) - q_sa
    return jnp.square(td)

n_step_loss_vmap = jax.vmap(n_step_loss, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)


@jax.jit
def td_error_batch(q_params, tgt_params, batch):
    # batch: states, actions, R, next_states, dones, gamma_pows
    q  = dqn_module.apply({"params": q_params}, batch[0])
    tq = dqn_module.apply({"params": tgt_params}, batch[3])
    return calculate_n_step_td_error_vmap(q, tq, batch[1], batch[2], batch[4], batch[5])


@jax.jit
def train_step(q_params, tgt_params, opt_state, batch):
    def loss_fn(p):
        q      = dqn_module.apply({"params": p}, batch[0])
        tq     = dqn_module.apply({"params": tgt_params}, batch[3])
        losses = n_step_loss_vmap(q, tq, batch[1], batch[2], batch[4], batch[5])
        return jnp.mean(losses)

    loss, grads        = jax.value_and_grad(loss_fn)(q_params)
    updates, opt_state = optimizer.update(grads, opt_state, q_params)
    q_params           = optax.apply_updates(q_params, updates)
    td_err             = td_error_batch(q_params, tgt_params, batch)
    return q_params, opt_state, loss, td_err


# Training
global_steps = 0
epsilon      = epsilon_max

try:
    for ep in range(num_episodes):
        state, info = env.reset()
        state       = np.array(state, dtype=np.float32)
        n_step_buffer.clear()
        ep_return   = 0.0

        while True:
            global_steps += 1

            # epsilon schedule
            if epsilon > epsilon_min:
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-epsilon_decay * global_steps)

            # act
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_vals = policy(q_params, jnp.asarray(state))
                action = int(action)

            # step
            next_state, reward, terminated, truncated, info = env.step(int(action))
            done       = bool(terminated or truncated)
            next_state = np.array(next_state, dtype=np.float32)

            ep_return += float(reward)

            # push into n-step buffer
            n_step_buffer.append((state, action, reward, next_state, done))

            # if buffer full, emit ONE fixed n-step sample from left
            if len(n_step_buffer) == n_steps:
                sample = make_n_step_sample(n_step_buffer, gamma, n_steps)
                td = td_error_single(q_params, tgt_params, sample)
                per_memory.add(td, sample)
                n_step_buffer.popleft()

            # if episode ended, flush remaining partial samples (k < n)
            if done:
                while len(n_step_buffer) > 0:
                    sample = make_n_step_sample(n_step_buffer, gamma, n_steps)
                    td = td_error_single(q_params, tgt_params, sample)
                    per_memory.add(td, sample)
                    n_step_buffer.popleft()

            # train
            if per_memory.size >= batch_size:
                batch = per_memory.sample(batch_size)
                idxs, data = zip(*batch)
                states, actions, rewards, next_states, dones, gamma_pows = zip(*data)

                q_params, opt_state, loss, new_td = train_step(
                    q_params, tgt_params, opt_state,
                    (
                        jnp.asarray(states                        ),
                        jnp.asarray(actions    , dtype=jnp.int32  ),
                        jnp.asarray(rewards    , dtype=jnp.float32),
                        jnp.asarray(next_states                   ),
                        jnp.asarray(dones      , dtype=jnp.float32),
                        jnp.asarray(gamma_pows , dtype=jnp.float32),
                    )
                )

                new_td_np = np.array(new_td)
                for i in range(batch_size):
                    per_memory.update(idxs[i], float(new_td_np[i]))

            # target sync
            if global_steps % sync_steps == 0:
                tgt_params = q_params

            if debug_render:
                env.render()

            state = next_state

            if done:
                print(f"{ep} - total reward: {ep_return:.0f}  eps: {epsilon:.3f}")
                break
finally:
    env.close()
