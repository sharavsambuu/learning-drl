#
# TRPO (Continuous, Squashed Gaussian) + GAE + CG(FVP) + KL-Capped Backtracking Line Search 
#
#   - Policy: Diagonal Gaussian in pre-squash space u ~ N(mean, std)
#   - Action: a = tanh(u) * max_action
#   - Buffer: Stores raw u (NOT action) for stable log-prob (no arctanh)
#   - Log-Prob: exact change-of-variables (u -> tanh(u) -> a), stable Jacobian term
#   - Trust Region: Analytic KL in u-space (Gaussian KL) averaged over on-policy states
#   - Proper Timeout Handling:
#       - dones_term = terminated only (controls bootstrapping in delta)
#       - masks      = continuing only (cuts GAE recursion on terminated OR truncated)
#   - CG Damping: ONLY for solving (F + damping I) x = g
#   - Step Scaling: Uses pure Fisher quadratic form x^T F x  (no damping in shs)
#   - Dense Reward Stability: advantage normalization (+ optional clip)
#   - Fixed-Batch Correctness: batch-edge bootstrap for GAE carry init
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


total_timesteps     = 500000
steps_per_batch     = 2048
value_updates       = 80

value_lr            = 3e-4
gamma               = 0.99
lambda_             = 0.95
kl_target           = 0.01

cg_iters            = 10
cg_damping          = 0.01

max_backtracks      = 15
backtrack_ratio     = 0.8
accept_ratio        = 0.1

max_grad_norm_v     = 0.5

debug_render        = True
render_update_every = 1
render_sleep        = 0.0

log_std_min         = -5.0
log_std_max         =  2.0


class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        init_ortho = nn.initializers.orthogonal(np.sqrt(2))
        init_mean  = nn.initializers.orthogonal(0.01)

        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)

        mean = nn.Dense(features=self.action_dim, kernel_init=init_mean)(x)

        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        log_std = jnp.clip(log_std, log_std_min, log_std_max)

        return mean, log_std

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


env        = gym.make("Pendulum-v1", render_mode="human" if debug_render else None)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy_module = PolicyNetwork(action_dim=action_dim)
value_module  = ValueNetwork()

rng = jax.random.PRNGKey(0)
rng, policy_rng, value_rng = jax.random.split(rng, 3)

dummy_input   = jnp.zeros((1, state_dim), dtype=jnp.float32)
policy_params = policy_module.init(policy_rng, dummy_input)["params"]
value_params  = value_module .init(value_rng , dummy_input)["params"]

value_optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm_v),
    optax.adam(value_lr, eps=1e-5)
)
value_opt_state = value_optimizer.init(value_params)


@jax.jit
def policy_inference(params, x):
    mean, log_std = policy_module.apply({"params": params}, x)
    log_std = jnp.broadcast_to(log_std, mean.shape)
    return mean, log_std

@jax.jit
def value_inference(params, x):
    return value_module.apply({"params": params}, x)

@jax.jit
def gaussian_log_prob(u, mean, log_std):
    var  = jnp.exp(2.0 * log_std)
    logp = -0.5 * jnp.sum(((u - mean) ** 2) / (var + 1e-10) + jnp.log(2.0 * jnp.pi * var + 1e-10), axis=-1)
    return logp

@jax.jit
def squashed_log_prob_from_u(u, mean, log_std):
    logp_u = gaussian_log_prob(u, mean, log_std)

    corr   = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    logdet = jnp.sum(jnp.log(max_action + 1e-10) + corr, axis=-1)

    return logp_u - logdet

@jax.jit
def sample_action(rng, params, obs):
    mean, log_std = policy_inference(params, obs)
    std = jnp.exp(log_std)

    u = mean + std * jax.random.normal(rng, shape=mean.shape, dtype=mean.dtype)
    a = jnp.tanh(u) * max_action

    return u[0], a[0]

@jax.jit
def gaussian_kl(old_mean, old_log_std, new_mean, new_log_std):
    old_var = jnp.exp(2.0 * old_log_std)
    new_var = jnp.exp(2.0 * new_log_std)

    num = old_var + (old_mean - new_mean) ** 2
    kl  = jnp.sum(new_log_std - old_log_std + 0.5 * (num / (new_var + 1e-10) - 1.0), axis=-1)
    return kl

@jax.jit
def calculate_kl_divergence(old_params, new_params, states):
    old_mean, old_log_std = policy_inference(old_params, states)
    new_mean, new_log_std = policy_inference(new_params, states)
    return jnp.mean(gaussian_kl(old_mean, old_log_std, new_mean, new_log_std))

@jax.jit
def compute_objective(new_params, old_params, states, raw_u, advantages):
    new_mean, new_log_std = policy_inference(new_params, states)
    old_mean, old_log_std = policy_inference(old_params, states)

    logp_new = squashed_log_prob_from_u(raw_u, new_mean, new_log_std)
    logp_old = squashed_log_prob_from_u(raw_u, old_mean, old_log_std)

    ratio = jnp.exp(logp_new - logp_old)
    return jnp.mean(ratio * advantages)

@jax.jit
def compute_advantages_returns(value_params, states, next_states, rewards, dones_term, masks):
    # dones_term: 1.0 only if TRUE terminal (terminated). Controls bootstrapping in delta.
    # masks     : 1.0 only if CONTINUING (not terminated and not truncated). Cuts GAE recursion.

    values      = value_inference(value_params, states).squeeze(-1)
    next_values = value_inference(value_params, next_states).squeeze(-1)

    deltas = rewards + gamma * (1.0 - dones_term) * next_values - values

    def scan_fn(carry, x):
        delta, mask = x
        carry = delta + gamma * lambda_ * mask * carry
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
def linesearch(old_params, full_step, states, raw_u, advantages, expected_improvement, max_backtracks=15):
    base_obj  = compute_objective(old_params, old_params, states, raw_u, advantages)
    step_tree = unflatten_params(full_step, old_params)

    def body_fun(carry, i):
        best_params, found = carry
        stepfrac = backtrack_ratio ** i

        test_params = jax.tree_util.tree_map(lambda p, s: p + stepfrac * s, old_params, step_tree)

        obj = compute_objective(test_params, old_params, states, raw_u, advantages)
        kl  = calculate_kl_divergence(old_params, test_params, states)

        improve_ok = obj >= base_obj + accept_ratio * stepfrac * expected_improvement
        kl_ok      = kl <= kl_target
        ok         = improve_ok & kl_ok

        best_params = jax.lax.cond(ok & (~found), lambda: test_params, lambda: best_params)
        found       = found | ok
        return (best_params, found), None

    (final_params, found), _ = jax.lax.scan(body_fun, (old_params, False), jnp.arange(max_backtracks))
    return jax.lax.cond(found, lambda: final_params, lambda: old_params)


updates      = int(total_timesteps // steps_per_batch)
global_steps = 0

try:
    for update in range(updates):
        states      = np.zeros((steps_per_batch, state_dim ), dtype=np.float32)
        next_states = np.zeros((steps_per_batch, state_dim ), dtype=np.float32)
        raw_u_buf   = np.zeros((steps_per_batch, action_dim), dtype=np.float32)
        actions     = np.zeros((steps_per_batch, action_dim), dtype=np.float32)
        rewards     = np.zeros((steps_per_batch,           ), dtype=np.float32)

        dones_term  = np.zeros((steps_per_batch,           ), dtype=np.float32)
        masks       = np.zeros((steps_per_batch,           ), dtype=np.float32)

        ep_returns = []
        ep_return  = 0.0

        obs, info = env.reset()
        obs = np.array(obs, dtype=np.float32)

        do_render = debug_render and (update % render_update_every == 0)

        for t in range(steps_per_batch):
            rng, a_rng = jax.random.split(rng)

            u, act = sample_action(a_rng, policy_params, jnp.array([obs], dtype=jnp.float32))
            act_np = np.array(act, dtype=np.float32)

            nobs, rew, terminated, truncated, info = env.step(act_np)
            nobs = np.array(nobs, dtype=np.float32)

            done_term = bool(terminated)
            mask_f    = 0.0 if (terminated or truncated) else 1.0

            states     [t] = obs
            next_states[t] = nobs
            raw_u_buf  [t] = np.array(u, dtype=np.float32)
            actions    [t] = act_np
            rewards    [t] = float(rew)

            dones_term [t] = 1.0 if done_term else 0.0
            masks      [t] = mask_f

            ep_return    += float(rew)
            global_steps += 1

            if do_render:
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
        raw_u_j       = jnp.asarray(raw_u_buf  , dtype=jnp.float32)
        rewards_j     = jnp.asarray(rewards    , dtype=jnp.float32)
        dones_term_j  = jnp.asarray(dones_term , dtype=jnp.float32)
        masks_j       = jnp.asarray(masks      , dtype=jnp.float32)

        advantages_j, returns_j = compute_advantages_returns(
            value_params, states_j, next_states_j, rewards_j, dones_term_j, masks_j
        )

        advantages_j = (advantages_j - jnp.mean(advantages_j)) / (jnp.std(advantages_j) + 1e-8)
        advantages_j = jnp.clip(advantages_j, -5.0, 5.0)

        v_loss = 0.0
        for _ in range(value_updates):
            value_params, value_opt_state, v_loss = value_update(value_params, value_opt_state, states_j, returns_j)

        old_params  = policy_params

        policy_grad = jax.grad(lambda p: compute_objective(p, old_params, states_j, raw_u_j, advantages_j))(old_params)
        flat_grad   = flatten_params(policy_grad)

        step_dir    = conjugate_gradient(old_params, states_j, flat_grad, max_iterations=cg_iters)

        fvp_step    = fisher_vector_product(step_dir, old_params, states_j)
        shs         = jnp.dot(step_dir, fvp_step)

        step_scale  = jnp.sqrt((2.0 * kl_target) / (shs + 1e-10))
        full_step   = step_dir * step_scale

        expected_improvement = jnp.dot(flat_grad, full_step)

        policy_params = linesearch(
            old_params,
            full_step,
            states_j,
            raw_u_j,
            advantages_j,
            expected_improvement,
            max_backtracks=max_backtracks
        )

        kl_val  = calculate_kl_divergence(old_params, policy_params, states_j)
        obj_old = compute_objective(old_params, old_params, states_j, raw_u_j, advantages_j)
        obj_new = compute_objective(policy_params, old_params, states_j, raw_u_j, advantages_j)

        avg_ep = float(np.mean(ep_returns)) if len(ep_returns) > 0 else float(ep_return)

        print(f"Update: {update:4d}, Steps: {global_steps:7d}, AvgEpRet: {avg_ep:8.1f}, "
              f"VLoss: {float(v_loss):.4f}, KL: {float(kl_val):.5f}, Obj: {float(obj_old):.4f}->{float(obj_new):.4f}")

finally:
    env.close()
