import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import jax.tree_util
from functools import partial


num_episodes  = 1000
learning_rate = 0.0003
gamma         = 0.99
kl_target     = 0.01
lambda_       = 0.95
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


env = gym.make('Pendulum-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

policy_module = PolicyNetwork(n_actions=n_actions)
value_module  = ValueNetwork()
dummy_input   = jnp.zeros((1, state_dim))
rng = jax.random.PRNGKey(0)
rng, policy_rng, value_rng = jax.random.split(rng, 3)
policy_params   = policy_module.init(policy_rng, dummy_input)['params']
value_params    = value_module.init(value_rng, dummy_input)['params']
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
    values = value_inference(value_params, states)
    next_values = jnp.concatenate([values[1:], jnp.zeros((1, 1))], axis=0)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    def discounted_advantages(carry, x):
        delta, done = x
        carry = carry * gamma * lambda_ * (1 - done) + delta
        return carry, carry
    _, advantages = jax.lax.scan(discounted_advantages, jnp.zeros_like(deltas), (deltas, dones), reverse=True)
    return advantages

@jax.jit
def value_update(value_params, value_opt_state, states, returns):
    def value_loss_fn(params):
        values = value_inference(params, states)
        return jnp.mean(jnp.square(values - returns))
    def body_fun(i, val):
        params, opt_state = val
        loss, grads = jax.value_and_grad(value_loss_fn)(params)
        updates, opt_state = value_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    value_params, value_opt_state = jax.lax.fori_loop(0, 50, body_fun, (value_params, value_opt_state))
    return value_params, value_opt_state, value_loss_fn(value_params)

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
    return -jnp.mean(surrogate_loss)

def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]

@jax.jit
def calculate_hvp_kl(old_params, states, p, damping_coeff=0.1):
    def kl_divergence_single_arg(params):
        return calculate_kl_divergence(old_params, params, states)
    unflat_p = unflatten_params(p, old_params)
    return flatten_params(hvp(kl_divergence_single_arg, (old_params,), (unflat_p,))) + damping_coeff * p

@jax.jit
def conjugate_gradient(old_params, states, b, num_iterations=10):
    x = jnp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = jnp.dot(r, r)
    for i in range(num_iterations):
        Ap = calculate_hvp_kl(old_params, states, p)
        alpha = rdotr / jnp.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = jnp.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

@partial(jax.jit, static_argnames=('max_backtracks',))
def linesearch(old_params, search_direction, states, actions, advantages, expected_improvement, max_backtracks=15, backtrack_ratio=0.8):
    def improvement_check(step_params):
        loss = compute_surrogate_loss(step_params, old_params, states, actions, advantages)
        improvement = compute_surrogate_loss(old_params, old_params, states, actions, advantages) - loss
        kl = calculate_kl_divergence(old_params, step_params, states)
        return improvement > 0, kl < kl_target
    def body_fun(carry, step):
        step_params, accepted = carry
        ratio = backtrack_ratio ** step
        new_step_params = jax.tree_util.tree_map(lambda p, u: p + ratio * u, old_params, unflatten_params(search_direction, old_params))
        imp_ok, kl_ok = improvement_check(new_step_params)
        step_params = jax.lax.cond(imp_ok & kl_ok, lambda: new_step_params, lambda: step_params)
        accepted = accepted | (imp_ok & kl_ok)
        return (step_params, accepted), None
    init_params = old_params
    (final_params, accepted), _ = jax.lax.scan(body_fun, (init_params, False), jnp.arange(max_backtracks))
    return jax.lax.cond(accepted, lambda: final_params, lambda: old_params)

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
            rng, action_rng = jax.random.split(rng)
            mean, log_std   = policy_inference(policy_params, jnp.array(state))
            std             = jnp.exp(log_std)
            action          = mean + std * jax.random.normal(action_rng, shape=mean.shape)
            action_np       = np.array(action)
            action_clipped  = np.clip(action_np, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, info = env.step(action_clipped)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward

            states .append(state         )
            actions.append(action_clipped)
            rewards.append(reward        )
            dones  .append(done          )
            state = next_state

        states  = jnp.array(states )
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        dones   = jnp.array(dones  )

        advantages = calculate_gae(value_params, states, rewards, dones)
        returns    = advantages + value_inference(value_params, states)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-6)
        advantages = jnp.clip(advantages, -2, 2)

        value_params, value_opt_state, value_loss = value_update(value_params, value_opt_state, states, returns)

        old_params           = policy_params
        b                    = flatten_params(jax.grad(compute_surrogate_loss)(old_params, old_params, states, actions, advantages))
        search_direction     = conjugate_gradient(old_params, states, b)
        expected_improvement = jnp.dot(b, search_direction)
        policy_params        = linesearch(old_params, search_direction, states, actions, advantages, expected_improvement)

        kl_divergence = calculate_kl_divergence(old_params, policy_params, states)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Value Loss: {value_loss:.4f}, KL: {kl_divergence:.4f}")

finally:
    env.close()