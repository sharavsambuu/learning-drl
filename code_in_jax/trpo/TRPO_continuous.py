import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import jax.tree_util

num_episodes  = 1000
learning_rate = 0.001
gamma         = 0.99
kl_target     = 0.01
batch_size    = 128


class PolicyNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        mean_output = nn.Dense(features=self.n_actions)(x)
        log_std = self.param('log_std', nn.initializers.constant(-0.5), (self.n_actions,))
        return mean_output, log_std

class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


env        = gym.make('Pendulum-v1', render_mode='human')
state_dim  = env.observation_space.shape[0]

if isinstance(env.action_space, gym.spaces.Discrete):
    n_actions = env.action_space.n
elif isinstance(env.action_space, gym.spaces.Box):
    n_actions = env.action_space.shape[0]
else:
    n_actions = 1

print(f"Action dimension : {n_actions}")

policy_module   = PolicyNetwork(n_actions=n_actions)
value_module    = ValueNetwork()
dummy_input     = jnp.zeros((1, state_dim))
policy_params   = policy_module.init(jax.random.PRNGKey(0), dummy_input)['params']
value_params    = value_module .init(jax.random.PRNGKey(1), dummy_input)['params']
value_optimizer = optax.adam(learning_rate)
value_opt_state = value_optimizer.init(value_params)


@jax.jit
def policy_inference(params, x):
    mean, log_std = policy_module.apply({'params': params}, x)
    return mean, log_std

@jax.jit
def value_inference(params, x):
    return value_module.apply({'params': params}, x)

@jax.jit
def gaussian_log_prob(actions, mean, log_std):
    std = jnp.exp(log_std)
    var = std ** 2
    log_prob = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var) + (actions - mean) ** 2 / var, axis=-1)
    return log_prob

@jax.jit
def calculate_gae(value_params, states, rewards, dones):
    values = value_module.apply({'params': value_params}, states)
    values = jnp.squeeze(values, axis=-1)
    next_values = jnp.concatenate([values[1:], jnp.zeros(1)], axis=0)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    def discounted_advantages(carry, delta_done):
        delta, done = delta_done
        return carry * gamma * (1 - done) + delta, carry * gamma * (1 - done) + delta
    _, advantages = jax.lax.scan(discounted_advantages, jnp.zeros_like(deltas[-1]), (deltas, dones), reverse=True)
    return advantages

@jax.jit
def value_update(value_params, value_opt_state, states, returns):
    def value_loss_fn(params):
        values = value_module.apply({'params': params}, states)
        return jnp.mean(jnp.square(values - returns))

    for _ in range(5):
        loss, grads = jax.value_and_grad(value_loss_fn)(value_params)
        updates, new_value_opt_state = value_optimizer.update(grads, value_opt_state, value_params)
        value_params = optax.apply_updates(value_params, updates)
    return value_params, new_value_opt_state, loss

@jax.jit
def calculate_kl_divergence(old_params, new_params, states):
    old_mean, old_log_std = policy_module.apply({'params': old_params}, states)
    new_mean, new_log_std = policy_module.apply({'params': new_params}, states)
    old_std = jnp.exp(old_log_std)
    new_std = jnp.exp(new_log_std)
    kl = jnp.mean(jnp.sum(jnp.log(new_std / old_std) + (old_std**2 + (old_mean - new_mean)**2) / (2 * new_std**2) - 0.5, axis=-1))
    return kl

@jax.jit
def compute_surrogate_loss(new_params, old_params, states, actions, advantages):
    new_mean, new_log_std = policy_module.apply({'params': new_params}, states)
    old_mean, old_log_std = policy_module.apply({'params': old_params}, states)
    log_prob_new = gaussian_log_prob(actions, new_mean, new_log_std)
    log_prob_old = gaussian_log_prob(actions, old_mean, old_log_std)
    ratio = jnp.exp(log_prob_new - log_prob_old)
    surrogate_loss = ratio * advantages
    return -jnp.mean(surrogate_loss)  # TRPO maximizes the surrogate objective, hence the negative sign

def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]

def calculate_hvp_kl(old_params, states, p, damping_coeff=0.1):
    def kl_divergence_single_arg(params):
        return calculate_kl_divergence(old_params, params, states)
    unflat_p = unflatten_params(p, old_params)
    # Add damping for numerical stability
    return flatten_params(hvp(kl_divergence_single_arg, (old_params,), (unflat_p,))) + damping_coeff * p

def conjugate_gradient(old_params, states, b, num_iterations=10):
    x = jnp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = jnp.dot(r, r)
    for _ in range(num_iterations):
        Ap = calculate_hvp_kl(old_params, states, p)
        alpha = rdotr / jnp.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = jnp.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def linesearch(old_params, new_params, states, actions, advantages, expected_improvement, max_backtracks=10, backtrack_ratio=0.8):
    for step in range(max_backtracks):
        ratio = backtrack_ratio ** step
        step_params = jax.tree.map(lambda p, u: p + ratio * u, old_params, unflatten_params(new_params, old_params))
        loss = compute_surrogate_loss(step_params, old_params, states, actions, advantages)
        improvement = compute_surrogate_loss(old_params, old_params, states, actions, advantages) - loss
        if improvement >= expected_improvement * ratio and improvement > 0:
            return step_params
    return old_params

def flatten_params(params):
    return jnp.concatenate([jnp.reshape(p, (-1,)) for p in jax.tree_util.tree_leaves(params)])

def unflatten_params(flat_params, params_example):
    shapes = [p.shape for p in jax.tree_util.tree_leaves(params_example)]
    sizes = [np.prod(s) for s in shapes]
    split_points = np.cumsum(sizes)[:-1]
    params_list = jnp.split(flat_params, split_points)
    new_params_list = [jnp.reshape(p, s) for p, s in zip(params_list, shapes)]
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_example), new_params_list)

try:
    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0

        while not done:
            mean, log_std = policy_inference(policy_params, jnp.array(state))
            std = jnp.exp(log_std)

            action = mean + std * jax.random.normal(jax.random.PRNGKey(episode), shape=mean.shape)  # Sample action from Gaussian policy

            action_np      = np.array(action)
            action_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, info = env.step(action_clipped)
            done           = terminated or truncated
            next_state     = np.array(next_state, dtype=np.float32)
            total_reward  += reward

            states .append(state         )
            actions.append(action_clipped)
            rewards.append(reward        )
            dones  .append(done          )
            state = next_state

        states  = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        dones   = jnp.array(dones)

        advantages = calculate_gae(value_params, states, rewards, dones)
        returns    = advantages + jnp.squeeze(value_inference(value_params, states), axis=-1)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-6)

        value_params, value_opt_state, value_loss = value_update(value_params, value_opt_state, states, returns)

        old_params       = policy_params
        b                = flatten_params(jax.grad(compute_surrogate_loss)(old_params, old_params, states, actions, advantages))
        search_direction = conjugate_gradient(old_params, states, b)
        expected_improvement = 0.5 * jnp.dot(b, search_direction)

        # Update policy parameters
        policy_params = linesearch(old_params, search_direction, states, actions, advantages, expected_improvement)

        # Calculate KL divergence after the update
        kl_divergence = calculate_kl_divergence(old_params, policy_params, states)
        if kl_divergence > 1.5 * kl_target:
            print(f'Episode {episode}: Early stopping due to high KL divergence: {kl_divergence}')

        print(f"Episode: {episode}, Total Reward: {total_reward}, Value Loss: {value_loss:.4f}, KL Divergence: {kl_divergence:.4f}")

finally:
    env.close()