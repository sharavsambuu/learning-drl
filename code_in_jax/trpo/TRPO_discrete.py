import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import jax.tree_util
from jax.scipy.sparse.linalg import cg as jax_cg
from functools import partial


num_episodes  = 5000
learning_rate = 0.003
gamma         = 0.99
kl_target     = 0.01
batch_size    = 128
n_actions     = 2


class PolicyNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        x = nn.softmax(x)
        return x

class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


env = gym.make('CartPole-v1', render_mode='human')

state_dim     = env.observation_space.shape[0]
policy_module = PolicyNetwork(n_actions=n_actions)
value_module  = ValueNetwork()

rng = jax.random.PRNGKey(0)
rng, policy_rng, value_rng = jax.random.split(rng, 3)
dummy_input     = jnp.zeros((1, state_dim))
policy_params   = policy_module.init(policy_rng, dummy_input)['params']
value_params    = value_module.init(value_rng, dummy_input)['params']
value_optimizer = optax.adam(learning_rate)
value_opt_state = value_optimizer.init(value_params)


@jax.jit
def policy_inference(params, x):
    return policy_module.apply({'params': params}, x)

@jax.jit
def value_inference(params, x):
    return value_module.apply({'params': params}, x)

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

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
    old_probs = policy_module.apply({'params': old_params}, states)
    new_probs = policy_module.apply({'params': new_params}, states)
    kl = jnp.sum(old_probs * (jnp.log(old_probs + 1e-10) - jnp.log(new_probs + 1e-10)), axis=-1)
    return kl

@jax.jit
def compute_surrogate_loss(new_params, old_params, states, actions, advantages):
    new_probs = policy_module.apply({'params': new_params}, states)
    old_probs = policy_module.apply({'params': old_params}, states)
    new_probs_for_actions = gather(new_probs, actions)
    old_probs_for_actions = gather(old_probs, actions)
    ratio = new_probs_for_actions / (old_probs_for_actions + 1e-10)
    
    entropy = -jnp.sum(new_probs * jnp.log(new_probs + 1e-10), axis=-1)
    entropy_bonus = 0.01 * jnp.mean(entropy)
    
    return jnp.mean(ratio * advantages) + entropy_bonus

def flatten_params(params):
    return jnp.concatenate([jnp.reshape(p, (-1,)) for p in jax.tree_util.tree_leaves(params)])

def unflatten_params(flat_params, params_example):
    shapes = [p.shape for p in jax.tree_util.tree_leaves(params_example)]
    sizes = [np.prod(s) for s in shapes]
    split_points = np.cumsum(sizes)[:-1]
    params_list = jnp.split(flat_params, split_points)
    new_params_list = [jnp.reshape(p, s) for p, s in zip(params_list, shapes)]
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_example), new_params_list)

def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]

@jax.jit
def fisher_vector_product(v, old_params, states):
    def kl_loss(params):
        return jnp.mean(calculate_kl_divergence(old_params, params, states))
    tangent = unflatten_params(v, old_params)
    return flatten_params(hvp(kl_loss, (old_params,), (tangent,)))

@partial(jax.jit, static_argnames=('max_iterations',))
def conjugate_gradient(old_params, states, b, max_iterations=15):
    def inner_loop(i, val):
        x, r, p, rdotr = val
        Ap = fisher_vector_product(p, old_params, states)
        alpha = rdotr / jnp.dot(p, Ap)
        x_next = x + alpha * p
        r_next = r - alpha * Ap
        rdotr_next = jnp.dot(r_next, r_next)
        beta = rdotr_next / (rdotr + 1e-10)
        p_next = r_next + beta * p
        return x_next, r_next, p_next, rdotr_next
    x = jnp.zeros_like(b)
    r = -b
    p = r
    rdotr = jnp.dot(r, r)
    x, r, p, rdotr = jax.lax.fori_loop(0, max_iterations, inner_loop, (x, r, p, rdotr))
    return x

@partial(jax.jit, static_argnames=('max_backtracks',))
def linesearch(old_params, new_params, states, actions, advantages, expected_improvement, max_backtracks=15, backtrack_ratio=0.8):
    def step_cond(loop_state):
        step, step_params = loop_state
        ratio = backtrack_ratio ** step
        step_params = jax.tree_util.tree_map(lambda p, u: p + ratio * u, old_params, unflatten_params(new_params, old_params))
        loss = compute_surrogate_loss(step_params, old_params, states, actions, advantages)
        improvement = compute_surrogate_loss(old_params, old_params, states, actions, advantages) - loss
        kl = jnp.max(calculate_kl_divergence(old_params, step_params, states))
        return jnp.all(jnp.array([kl > kl_target, improvement < 0.5 * expected_improvement * ratio, step < max_backtracks]))
    def step_body(loop_state):
        step, step_params = loop_state
        return (step + 1, step_params)
    step        = 0
    step_params = old_params
    init_val    = (step, step_params)
    final_step, final_step_params = jax.lax.while_loop(step_cond, step_body, init_val)
    loss = compute_surrogate_loss(final_step_params, old_params, states, actions, advantages)
    improvement = compute_surrogate_loss(old_params, old_params, states, actions, advantages) - loss
    kl = jnp.max(calculate_kl_divergence(old_params, final_step_params, states))
    step_params = jax.lax.cond(
        jnp.all(jnp.array([kl <= kl_target, improvement >= 0.5 * expected_improvement * backtrack_ratio**final_step])),
        lambda: final_step_params,
        lambda: old_params
    )
    return step_params



try:
    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], []
        state, info  = env.reset()
        state        = np.array(state, dtype=np.float32)
        done         = False
        total_reward = 0

        while not done:
            rng, action_rng = jax.random.split(rng)
            action_probs    = policy_inference(policy_params, jnp.array([state]))
            action          = jax.random.categorical(action_rng, action_probs).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done          = terminated or truncated
            next_state    = np.array(next_state, dtype=np.float32)
            total_reward += reward

            states .append(state )
            actions.append(action)
            rewards.append(reward)
            dones  .append(done  )

            state = next_state

        states  = jnp.array(states )
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        dones   = jnp.array(dones  )

        advantages = calculate_gae(value_params, states, rewards, dones)
        returns    = advantages + jnp.squeeze(value_inference(value_params, states), axis=-1)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-10)
        advantages = jnp.clip(advantages, -5, 5)

        value_params, value_opt_state, value_loss = value_update(value_params, value_opt_state, states, returns)

        old_params           = policy_params
        policy_grad          = jax.grad(compute_surrogate_loss)(old_params, old_params, states, actions, advantages)
        flat_grad            = flatten_params(policy_grad)
        grad_norm            = jnp.linalg.norm(flat_grad)
        search_direction     = conjugate_gradient(old_params, states, flat_grad)
        expected_improvement = jnp.dot(search_direction, flat_grad)
        policy_params        = linesearch(old_params, search_direction, states, actions, advantages, expected_improvement)

        kl_divergence        = jnp.max(calculate_kl_divergence(old_params, policy_params, states))

        print(f"Episode: {episode}, Total Reward: {total_reward}, "
              f"Value Loss: {value_loss:.4f}, KL Divergence: {kl_divergence:.4f}, "
              f"Grad Norm: {grad_norm:.4f}")

finally:
    env.close()