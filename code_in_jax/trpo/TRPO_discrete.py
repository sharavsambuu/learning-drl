#
# TRPO (Discrete) + GAE + CG(FVP) + KL-Capped Backtracking Line Search + Fixed-Batch
#

import time
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import jax.tree_util
from functools import partial

# Hyperparameters
total_timesteps     = 200000
steps_per_batch     = 1024
value_updates       = 25

learning_rate       = 1e-3
gamma               = 0.99
lambda_             = 0.95
kl_target           = 0.01

cg_iters            = 10
cg_damping          = 0.01  # Damping helps CG stability

max_backtracks      = 15
backtrack_ratio     = 0.8
accept_ratio        = 0.1

debug_render        = True 


# Network Definitions

class PolicyNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.n_actions)(x)
        x = nn.softmax(x)
        return x

class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1)(x)
        return x


# Setup
env           = gym.make('CartPole-v1', render_mode="human" if debug_render else None)
state_dim     = env.observation_space.shape[0]
n_actions     = env.action_space.n

policy_module = PolicyNetwork(n_actions=n_actions)
value_module  = ValueNetwork()

rng = jax.random.PRNGKey(42)
rng, policy_rng, value_rng = jax.random.split(rng, 3)

dummy_input     = jnp.zeros((1, state_dim), dtype=jnp.float32)
policy_params   = policy_module.init(policy_rng, dummy_input)['params']
value_params    = value_module .init(value_rng , dummy_input)['params']

value_optimizer = optax.adam(learning_rate)
value_opt_state = value_optimizer.init(value_params)


# JAX Functions

@jax.jit
def policy_inference(params, x):
    return policy_module.apply({'params': params}, x)

@jax.jit
def value_inference(params, x):
    return value_module.apply({'params': params}, x)

@jax.jit
def calculate_kl_divergence(old_params, new_params, states):
    old_probs = policy_inference(old_params, states)
    new_probs = policy_inference(new_params, states)
    # Sum over actions, mean over batch
    kl = jnp.sum(old_probs * (jnp.log(old_probs + 1e-10) - jnp.log(new_probs + 1e-10)), axis=-1)
    return jnp.mean(kl)

@jax.jit
def compute_objective(new_params, old_params, states, actions, advantages):
    new_probs = policy_inference(new_params, states)
    old_probs = policy_inference(old_params, states)

    actions    = actions.astype(jnp.int32).reshape(-1, 1)
    advantages = jnp.reshape(advantages, (-1,))

    new_pa = jnp.take_along_axis(new_probs, actions, axis=1).squeeze(-1)
    old_pa = jnp.take_along_axis(old_probs, actions, axis=1).squeeze(-1)

    ratio = new_pa / (old_pa + 1e-10)
    return jnp.mean(ratio * advantages)

@jax.jit
def compute_advantages_returns(value_params, states, next_states, rewards, dones):
    # Flatten inputs to (Batch,)
    values      = value_inference(value_params, states).squeeze(-1)
    next_values = value_inference(value_params, next_states).squeeze(-1)

    # Standard GAE Delta
    deltas = rewards + gamma * (1.0 - dones) * next_values - values

    def scan_fn(carry, x):
        delta, done = x
        # If done=1, previous advantage (carry) is cut off.
        carry = delta + gamma * lambda_ * (1.0 - done) * carry
        return carry, carry

    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.array(0.0, dtype=deltas.dtype), # Assume 0 advantage after batch/episode end
        (deltas, dones),
        reverse=True
    )

    returns = advantages + values
    return advantages, returns

@jax.jit
def value_update(value_params, value_opt_state, states, targets):
    def loss_fn(params):
        preds = value_inference(params, states).squeeze(-1)
        return jnp.mean(jnp.square(preds - targets))

    def body_fun(i, val):
        params, opt_state = val
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = value_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    value_params, value_opt_state = jax.lax.fori_loop(0, value_updates, body_fun, (value_params, value_opt_state))
    loss = loss_fn(value_params)
    return value_params, value_opt_state, loss

# Flatten/Unflatten Utils
def flatten_params(params):
    return jnp.concatenate([jnp.reshape(p, (-1,)) for p in jax.tree_util.tree_leaves(params)])

def unflatten_params(flat_params, params_example):
    leaves  = jax.tree_util.tree_leaves(params_example)
    shapes  = [p.shape for p in leaves]
    sizes   = [int(np.prod(s)) for s in shapes]
    split_points = np.cumsum(sizes)[:-1].tolist()
    parts = jnp.split(flat_params, split_points) if len(split_points) > 0 else [flat_params]
    rebuilt = [p.reshape(s) for p, s in zip(parts, shapes)]
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_example), rebuilt)

# Fisher & CG
@jax.jit
def fisher_vector_product(v, old_params, states):
    def kl_term(p):
        return calculate_kl_divergence(old_params, p, states)
    
    tangent = unflatten_params(v, old_params)
    # Forward-over-Reverse AD
    hv = jax.jvp(jax.grad(kl_term), (old_params,), (tangent,))[1]
    return flatten_params(hv)

@partial(jax.jit, static_argnames=('max_iterations',))
def conjugate_gradient(old_params, states, b, max_iterations=10):
    def body(i, val):
        x, r, p, rdotr = val
        Ap = fisher_vector_product(p, old_params, states)
        Ap = Ap + cg_damping * p # Damping for matrix inversion stability
        alpha = rdotr / (jnp.dot(p, Ap) + 1e-10)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        new_rdotr = jnp.dot(r_new, r_new)
        beta = new_rdotr / (rdotr + 1e-10)
        p_new = r_new + beta * p
        return x_new, r_new, p_new, new_rdotr

    x = jnp.zeros_like(b)
    r = b
    p = b
    rdotr = jnp.dot(r, r)
    x, _, _, _ = jax.lax.fori_loop(0, max_iterations, body, (x, r, p, rdotr))
    return x

@partial(jax.jit, static_argnames=('max_backtracks',))
def linesearch(old_params, full_step, states, actions, advantages, expected_improvement, max_backtracks=15):
    base_obj   = compute_objective(old_params, old_params, states, actions, advantages)
    step_tree  = unflatten_params(full_step, old_params)

    def body_fun(carry, i):
        best_params, found = carry
        stepfrac = backtrack_ratio ** i

        test_params = jax.tree_util.tree_map(lambda p, s: p + stepfrac * s, old_params, step_tree)

        obj = compute_objective(test_params, old_params, states, actions, advantages)
        kl  = calculate_kl_divergence(old_params, test_params, states)

        # Armijo condition (approx) + KL Constraint
        improve_ok = obj >= base_obj + accept_ratio * stepfrac * expected_improvement
        kl_ok      = kl <= kl_target
        ok         = improve_ok & kl_ok

        # Update best_params if ok and not found yet
        best_params = jax.lax.cond(ok & (~found), lambda: test_params, lambda: best_params)
        found       = found | ok
        return (best_params, found), None

    (final_params, found), _ = jax.lax.scan(body_fun, (old_params, False), jnp.arange(max_backtracks))
    return jax.lax.cond(found, lambda: final_params, lambda: old_params)


# Main Loop

updates = int(total_timesteps // steps_per_batch)
global_steps = 0

try:
    for update in range(updates):
        # Pre-allocate buffers
        states      = np.zeros((steps_per_batch, state_dim), dtype=np.float32)
        next_states = np.zeros((steps_per_batch, state_dim), dtype=np.float32)
        actions     = np.zeros((steps_per_batch,          ), dtype=np.int32  )
        rewards     = np.zeros((steps_per_batch,          ), dtype=np.float32)
        dones       = np.zeros((steps_per_batch,          ), dtype=np.float32)

        ep_returns = []
        ep_return  = 0.0

        obs, info = env.reset()
        obs = np.array(obs, dtype=np.float32)

        for t in range(steps_per_batch):
            rng, a_rng = jax.random.split(rng)
            
            # Policy Inference (CPU -> GPU -> CPU)
            probs = policy_inference(policy_params, jnp.array([obs], dtype=jnp.float32))[0]
            act   = int(jax.random.categorical(a_rng, jnp.log(probs + 1e-10)).item())

            nobs, rew, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)
            nobs = np.array(nobs, dtype=np.float32)

            states     [t] = obs
            next_states[t] = nobs
            actions    [t] = act
            rewards    [t] = float(rew)
            dones      [t] = 1.0 if done else 0.0

            ep_return += float(rew)
            global_steps += 1

            if debug_render:
                env.render()

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                obs, info = env.reset()
                obs = np.array(obs, dtype=np.float32)
            else:
                obs = nobs

        # Update Phase
        states_j      = jnp.asarray(states     , dtype=jnp.float32)
        next_states_j = jnp.asarray(next_states, dtype=jnp.float32)
        actions_j     = jnp.asarray(actions    , dtype=jnp.int32  )
        rewards_j     = jnp.asarray(rewards    , dtype=jnp.float32)
        dones_j       = jnp.asarray(dones      , dtype=jnp.float32)

        # GAE & Value Update
        advantages_j, returns_j = compute_advantages_returns(value_params, states_j, next_states_j, rewards_j, dones_j)
        
        # Normalize advantages
        advantages_j = (advantages_j - jnp.mean(advantages_j)) / (jnp.std(advantages_j) + 1e-6)
        advantages_j = jnp.clip(advantages_j, -5.0, 5.0)

        value_params, value_opt_state, value_loss = value_update(value_params, value_opt_state, states_j, returns_j)

        # TRPO Update
        old_params = policy_params

        # Get Gradient (Ascent direction)
        policy_grad = jax.grad(lambda p: compute_objective(p, old_params, states_j, actions_j, advantages_j))(old_params)
        flat_grad   = flatten_params(policy_grad)

        # Conjugate Gradient
        step_dir = conjugate_gradient(old_params, states_j, flat_grad, max_iterations=cg_iters)

        # Measure FVP for step scaling
        fvp_step = fisher_vector_product(step_dir, old_params, states_j)
        
        # pure FVP for KL constraint, ignoring damping here
        shs = jnp.dot(step_dir, fvp_step) 
        
        # Avoid divide by zero
        max_beta   = jnp.sqrt((2.0 * kl_target) / (shs + 1e-10))
        full_step  = step_dir * max_beta

        expected_improvement = jnp.dot(flat_grad, full_step)

        # Line Search
        policy_params = linesearch(old_params, full_step, states_j, actions_j, advantages_j, expected_improvement, max_backtracks=max_backtracks)

        # Metrics
        kl_val  = calculate_kl_divergence(old_params, policy_params, states_j)
        obj_old = compute_objective(old_params, old_params, states_j, actions_j, advantages_j)
        obj_new = compute_objective(policy_params, old_params, states_j, actions_j, advantages_j)

        avg_ep = float(np.mean(ep_returns)) if len(ep_returns) > 0 else float(ep_return)

        print(f"Update: {update:4d}, Steps: {global_steps:7d}, AvgEpRet: {avg_ep:7.1f}, "
              f"VLoss: {float(value_loss):.4f}, KL: {float(kl_val):.5f}, Obj: {float(obj_old):.4f}->{float(obj_new):.4f}")

finally:
    env.close()