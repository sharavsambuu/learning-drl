#
# TRPO (Discrete)
#
#   - Policy: Logits-based Categorical (stable log-softmax)
#   - Critic: Baseline Value Function (MSE regression)
#   - Advantage: GAE(λ) + Advantage Normalization (dense reward stability)
#   - Proper Timeout Handling:
#       - dones_term = terminated only (controls bootstrapping)
#       - masks      = continuing only (cuts GAE recursion on terminated OR truncated)
#   - Natural Gradient Step:
#       - Fisher-Vector Product (KL Hessian) via JVP(grad(KL))
#       - Conjugate Gradient (CG) solve with damping: (F + damping * I)^-1 g
#   - Trust Region Enforcement:
#       - KL-capped step scaling using damped curvature (safe/conservative)
#       - Backtracking line search with (1) KL constraint and (2) improvement test
#   - Fixed-batch update (one TRPO step per rollout batch)
#
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
steps_per_batch     = 2048

gamma               = 0.99
gae_lambda          = 0.95

value_lr            = 1e-3
value_updates       = 25

kl_target           = 0.01

cg_iters            = 10
cg_damping          = 0.01

max_backtracks      = 15
backtrack_ratio     = 0.8
accept_ratio        = 0.1

max_grad_norm_v     = 0.5     # critic only 

debug_render        = True
render_sleep        = 0.0


# Network Definitions

class PolicyNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        init_ortho = nn.initializers.orthogonal(np.sqrt(2))
        init_out   = nn.initializers.orthogonal(0.01)

        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)

        logits = nn.Dense(features=self.n_actions, kernel_init=init_out)(x)
        return logits

class ValueNetwork(nn.Module):

    @nn.compact
    def __call__(self, x):
        init_ortho = nn.initializers.orthogonal(np.sqrt(2))
        init_out   = nn.initializers.orthogonal(1.0)

        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)

        v = nn.Dense(features=1, kernel_init=init_out)(x)
        return v


# Setup

env       = gym.make("CartPole-v1", render_mode="human" if debug_render else None)
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_module = PolicyNetwork(n_actions=n_actions)
value_module  = ValueNetwork()

rng = jax.random.PRNGKey(42)
rng, p_rng, v_rng = jax.random.split(rng, 3)

dummy_input   = jnp.zeros((1, state_dim), dtype=jnp.float32)
policy_params = policy_module.init(p_rng, dummy_input)["params"]
value_params  = value_module .init(v_rng, dummy_input)["params"]

value_optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm_v),
    optax.adam(value_lr, eps=1e-5)
)
value_opt_state = value_optimizer.init(value_params)


# JAX Functions

@jax.jit
def policy_logits(params, x):
    return policy_module.apply({"params": params}, x)

@jax.jit
def value_inference(params, x):
    return value_module.apply({"params": params}, x)

@jax.jit
def logprob_from_logits(logits, actions):
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    actions  = actions.astype(jnp.int32).reshape(-1, 1)
    return jnp.take_along_axis(logp_all, actions, axis=1).squeeze(-1)

@jax.jit
def calculate_kl_divergence(old_params, new_params, states):
    old_logits = policy_logits(old_params, states)
    new_logits = policy_logits(new_params, states)

    old_logp   = jax.nn.log_softmax(old_logits, axis=-1)
    new_logp   = jax.nn.log_softmax(new_logits, axis=-1)

    old_p      = jnp.exp(old_logp)
    kl         = jnp.sum(old_p * (old_logp - new_logp), axis=-1)
    return jnp.mean(kl)

@jax.jit
def compute_objective(new_params, old_params, states, actions, advantages):
    new_logits = policy_logits(new_params, states)
    old_logits = policy_logits(old_params, states)

    new_logpa  = logprob_from_logits(new_logits, actions)
    old_logpa  = logprob_from_logits(old_logits, actions)

    ratio      = jnp.exp(new_logpa - old_logpa)
    adv        = jnp.reshape(advantages, (-1,))
    return jnp.mean(ratio * adv)

@jax.jit
def compute_advantages_returns(value_params, states, next_states, rewards, dones_term, masks):
    values      = value_inference(value_params, states).squeeze(-1)
    next_values = value_inference(value_params, next_states).squeeze(-1)

    deltas = rewards + gamma * (1.0 - dones_term) * next_values - values

    def scan_fn(carry, x):
        delta, mask = x
        carry = delta + gamma * gae_lambda * mask * carry
        return carry, carry

    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.array(0.0, dtype=deltas.dtype),
        (deltas, masks),
        reverse=True
    )

    returns = advantages + values
    return advantages, returns

@jax.jit
def value_update(value_params, value_opt_state, states, targets):
    def loss_fn(params):
        preds = value_inference(params, states).squeeze(-1)
        return jnp.mean(jnp.square(preds - targets))

    loss, grads = jax.value_and_grad(loss_fn)(value_params)
    updates, value_opt_state = value_optimizer.update(grads, value_opt_state, value_params)
    value_params = optax.apply_updates(value_params, updates)
    return value_params, value_opt_state, loss


# Flatten/Unflatten Utils

def flatten_params(params):
    return jnp.concatenate([jnp.reshape(p, (-1,)) for p in jax.tree_util.tree_leaves(params)])

def unflatten_params(flat_params, params_example):
    leaves       = jax.tree_util.tree_leaves(params_example)
    shapes       = [p.shape for p in leaves]
    sizes        = [int(np.prod(s)) for s in shapes]
    split_points = np.cumsum(sizes)[:-1].tolist()
    parts        = jnp.split(flat_params, split_points) if len(split_points) > 0 else [flat_params]
    rebuilt      = [p.reshape(s) for p, s in zip(parts, shapes)]
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_example), rebuilt)


# Fisher & CG

@jax.jit
def fisher_vector_product(v, old_params, states):
    def kl_term(p):
        return calculate_kl_divergence(old_params, p, states)

    tangent = unflatten_params(v, old_params)
    hv = jax.jvp(jax.grad(kl_term), (old_params,), (tangent,))[1]
    return flatten_params(hv)

@partial(jax.jit, static_argnames=("max_iterations",))
def conjugate_gradient(old_params, states, b, max_iterations=10):
    def body(i, val):
        x, r, p, rdotr = val
        Ap        = fisher_vector_product(p, old_params, states)
        Ap        = Ap + cg_damping * p
        alpha     = rdotr / (jnp.dot(p, Ap) + 1e-10)
        x_new     = x + alpha * p
        r_new     = r - alpha * Ap
        new_rdotr = jnp.dot(r_new, r_new)
        beta      = new_rdotr / (rdotr + 1e-10)
        p_new     = r_new + beta * p
        return x_new, r_new, p_new, new_rdotr

    x = jnp.zeros_like(b)
    r = b
    p = b
    rdotr = jnp.dot(r, r)
    x, _, _, _ = jax.lax.fori_loop(0, max_iterations, body, (x, r, p, rdotr))
    return x

@partial(jax.jit, static_argnames=("max_backtracks",))
def linesearch(old_params, full_step, states, actions, advantages, expected_improvement, max_backtracks=15):
    base_obj  = compute_objective(old_params, old_params, states, actions, advantages)
    step_tree = unflatten_params(full_step, old_params)

    def body_fun(carry, i):
        best_params, found = carry
        stepfrac = backtrack_ratio ** i

        test_params = jax.tree_util.tree_map(lambda p, s: p + stepfrac * s, old_params, step_tree)

        obj = compute_objective(test_params, old_params, states, actions, advantages)
        kl  = calculate_kl_divergence(old_params, test_params, states)

        improve_ok = obj >= base_obj + accept_ratio * stepfrac * expected_improvement
        kl_ok      = kl <= kl_target
        ok         = improve_ok & kl_ok

        best_params = jax.lax.cond(ok & (~found), lambda: test_params, lambda: best_params)
        found       = found | ok
        return (best_params, found), None

    (final_params, found), _ = jax.lax.scan(body_fun, (old_params, False), jnp.arange(max_backtracks))
    return jax.lax.cond(found, lambda: final_params, lambda: old_params)


# Main Loop

updates      = int(total_timesteps // steps_per_batch)
global_steps = 0

try:
    for update in range(updates):
        states      = np.zeros((steps_per_batch, state_dim), dtype=np.float32)
        next_states = np.zeros((steps_per_batch, state_dim), dtype=np.float32)
        actions     = np.zeros((steps_per_batch,          ), dtype=np.int32  )
        rewards     = np.zeros((steps_per_batch,          ), dtype=np.float32)

        dones_term  = np.zeros((steps_per_batch,          ), dtype=np.float32)  # terminated only
        masks       = np.zeros((steps_per_batch,          ), dtype=np.float32)  # continue only (cuts on term OR trunc)

        ep_returns  = []
        ep_return   = 0.0

        obs, info = env.reset()
        obs = np.array(obs, dtype=np.float32)

        for t in range(steps_per_batch):
            rng, a_rng = jax.random.split(rng)

            logits = policy_logits(policy_params, jnp.array([obs], dtype=jnp.float32))[0]
            act    = int(jax.random.categorical(a_rng, logits).item())

            nobs, rew, terminated, truncated, info = env.step(act)
            nobs = np.array(nobs, dtype=np.float32)

            done_term = bool(terminated)
            mask_f    = 0.0 if (terminated or truncated) else 1.0

            states     [t] = obs
            next_states[t] = nobs
            actions    [t] = act
            rewards    [t] = float(rew)

            dones_term [t] = 1.0 if done_term else 0.0
            masks      [t] = mask_f

            ep_return   += float(rew)
            global_steps += 1

            if debug_render:
                env.render()
                if render_sleep > 0:
                    time.sleep(render_sleep)

            if terminated or truncated:
                ep_returns.append(ep_return)
                ep_return = 0.0
                obs, info = env.reset()
                obs = np.array(obs, dtype=np.float32)
            else:
                obs = nobs

        states_j      = jnp.asarray(states     , dtype=jnp.float32)
        next_states_j = jnp.asarray(next_states, dtype=jnp.float32)
        actions_j     = jnp.asarray(actions    , dtype=jnp.int32  )
        rewards_j     = jnp.asarray(rewards    , dtype=jnp.float32)
        dones_term_j  = jnp.asarray(dones_term , dtype=jnp.float32)
        masks_j       = jnp.asarray(masks      , dtype=jnp.float32)

        # GAE (policy) + Returns (critic)
        advantages_j, returns_j = compute_advantages_returns(
            value_params, states_j, next_states_j, rewards_j, dones_term_j, masks_j
        )

        advantages_j = (advantages_j - jnp.mean(advantages_j)) / (jnp.std(advantages_j) + 1e-6)
        advantages_j = jnp.clip(advantages_j, -5.0, 5.0)

        # Critic Fit (multiple SGD steps)
        v_loss = 0.0
        for _ in range(value_updates):
            value_params, value_opt_state, v_loss = value_update(value_params, value_opt_state, states_j, returns_j)

        # TRPO Policy Update
        old_params = policy_params

        policy_grad = jax.grad(
            lambda p: compute_objective(p, old_params, states_j, actions_j, advantages_j)
        )(old_params)

        flat_grad = flatten_params(policy_grad)

        step_dir  = conjugate_gradient(old_params, states_j, flat_grad, max_iterations=cg_iters)

        fvp_step  = fisher_vector_product(step_dir, old_params, states_j) + cg_damping * step_dir
        shs       = jnp.dot(step_dir, fvp_step)

        max_beta  = jnp.sqrt((2.0 * kl_target) / (shs + 1e-10))
        full_step = step_dir * max_beta

        expected_improvement = jnp.dot(flat_grad, full_step)

        policy_params = linesearch(
            old_params,
            full_step,
            states_j,
            actions_j,
            advantages_j,
            expected_improvement,
            max_backtracks=max_backtracks
        )

        kl_val  = calculate_kl_divergence(old_params, policy_params, states_j)
        obj_old = compute_objective(old_params, old_params, states_j, actions_j, advantages_j)
        obj_new = compute_objective(policy_params, old_params, states_j, actions_j, advantages_j)

        avg_ep = float(np.mean(ep_returns)) if len(ep_returns) > 0 else float(ep_return)

        print(f"Update: {update:4d}, Steps: {global_steps:7d}, AvgEpRet: {avg_ep:7.1f}, "
              f"VLoss: {float(v_loss):.4f}, KL: {float(kl_val):.5f}, Obj: {float(obj_old):.4f}->{float(obj_new):.4f}")

finally:
    env.close()
