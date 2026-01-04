#
# PPO (Discrete)
#
# - Orthogonal Init 
# - Gradient Clipping 
# - GAE(λ)
# - Proper Truncation Handling (bootstrap) 
# - Proper Episode Cut (GAE recursion)
# - Value Clipping (PPO paper) 
# - LR Annealing 
# - KL Early Stop
#
#

import time
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


# Hyperparameters
total_timesteps     = 500000
steps_per_batch     = 2048
learning_rate       = 3e-4

gamma               = 0.99
gae_lambda          = 0.95

clip_epsilon        = 0.2
entropy_coeff       = 0.001
value_coeff         = 0.5
max_grad_norm       = 0.5

epochs_per_update   = 10
mini_batch_size     = 256

target_kl           = 0.01     # KL early stop threshold (0.01 ~ 0.03)

debug_render        = True
render_update_every = 10
render_sleep        = 0.0


class ActorCriticNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        init_ortho = nn.initializers.orthogonal(np.sqrt(2))
        init_layer = nn.initializers.orthogonal(0.01)
        init_value = nn.initializers.orthogonal(1.0)

        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)

        logits = nn.Dense(features=self.n_actions, kernel_init=init_layer)(x)
        value  = nn.Dense(features=1,              kernel_init=init_value)(x)

        return logits, value


env       = gym.make("CartPole-v1", render_mode="human" if debug_render else None)
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

actor_critic_module = ActorCriticNetwork(n_actions=n_actions)

rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng, 2)

dummy_input           = jnp.zeros((1, state_dim), dtype=jnp.float32)
actor_critic_params   = actor_critic_module.init(init_rng, dummy_input)["params"]

updates               = int(total_timesteps // steps_per_batch)
minibatches_per_epoch = steps_per_batch // mini_batch_size
total_opt_steps       = updates * epochs_per_update * minibatches_per_epoch

lr_schedule = optax.linear_schedule(
    init_value       = learning_rate,
    end_value        = 0.0,
    transition_steps = total_opt_steps
)

optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(lr_schedule, eps=1e-5)
)
opt_state = optimizer.init(actor_critic_params)


@jax.jit
def ac_inference(params, x):
    logits, value = actor_critic_module.apply({"params": params}, x)
    return logits, value

@jax.jit
def logprob_from_logits(logits, actions):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions   = actions.reshape(-1, 1)
    return jnp.take_along_axis(log_probs, actions, axis=1).squeeze(-1)

@jax.jit
def entropy_from_logits(logits):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs     = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1).mean()

@jax.jit
def calculate_gae(rewards, values, next_values, dones_term, masks):
    # dones_term: 1.0 only if TRUE terminal (terminated). Controls bootstrapping.
    # masks     : 1.0 only if CONTINUING (not terminated and not truncated). Cuts recursion at episode boundary.

    deltas = rewards + gamma * (1.0 - dones_term) * next_values - values

    def scan_fn(carry, x):
        delta, mask = x
        carry = delta + gamma * gae_lambda * mask * carry
        return carry, carry

    _, adv = jax.lax.scan(
        scan_fn,
        jnp.array(0.0, dtype=deltas.dtype),
        (deltas, masks),
        reverse=True
    )

    returns = adv + values
    return adv, returns

@jax.jit
def ppo_loss(params, states, actions, old_log_probs, advantages, returns, old_values):
    logits, values = ac_inference(params, states)
    values = values.squeeze(-1)

    new_log_probs = logprob_from_logits(logits, actions)
    ratio         = jnp.exp(new_log_probs - old_log_probs)

    pg_loss1      = -advantages * ratio
    pg_loss2      = -advantages * jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    actor_loss    = jnp.maximum(pg_loss1, pg_loss2).mean()

    v_clipped     = old_values + jnp.clip(values - old_values, -clip_epsilon, clip_epsilon)
    v_loss1       = jnp.square(returns - values)
    v_loss2       = jnp.square(returns - v_clipped)
    critic_loss   = jnp.maximum(v_loss1, v_loss2).mean()

    entropy       = entropy_from_logits(logits)
    approx_kl     = (old_log_probs - new_log_probs).mean()
    clipfrac      = (jnp.abs(ratio - 1.0) > clip_epsilon).mean()

    total_loss    = actor_loss + value_coeff * critic_loss - entropy_coeff * entropy
    return total_loss, (actor_loss, critic_loss, entropy, approx_kl, clipfrac)

@jax.jit
def train_step(params, opt_state, states, actions, old_log_probs, advantages, returns, old_values):
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, (actor_loss, critic_loss, entropy, approx_kl, clipfrac)), grads = grad_fn(
        params, states, actions, old_log_probs, advantages, returns, old_values
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, actor_loss, critic_loss, entropy, approx_kl, clipfrac


global_steps = 0

try:
    for update in range(updates):
        states      = np.zeros((steps_per_batch, state_dim), dtype=np.float32)
        actions     = np.zeros((steps_per_batch,), dtype=np.int32  )
        rewards     = np.zeros((steps_per_batch,), dtype=np.float32)

        dones_term  = np.zeros((steps_per_batch,), dtype=np.float32)  # terminated only
        masks       = np.zeros((steps_per_batch,), dtype=np.float32)  # continue only (cuts on term OR trunc)

        values      = np.zeros((steps_per_batch,), dtype=np.float32)
        next_values = np.zeros((steps_per_batch,), dtype=np.float32)
        log_probs   = np.zeros((steps_per_batch,), dtype=np.float32)

        ep_returns = []
        ep_return  = 0.0

        obs, info = env.reset()
        obs = np.array(obs, dtype=np.float32)

        do_render = debug_render and (update % render_update_every == 0)

        for t in range(steps_per_batch):
            rng, a_rng = jax.random.split(rng)

            logits, v = ac_inference(actor_critic_params, jnp.array([obs], dtype=jnp.float32))
            logits = logits[0]
            v      = float(v[0, 0])

            action = jax.random.categorical(a_rng, logits).astype(jnp.int32)
            logp   = logprob_from_logits(logits.reshape(1, -1), action.reshape(1,))
            logp   = float(logp[0])

            nobs, rew, terminated, truncated, info = env.step(int(action))
            nobs = np.array(nobs, dtype=np.float32)

            if terminated:
                v_next = 0.0
            else:
                _, v_n = ac_inference(actor_critic_params, jnp.array([nobs], dtype=jnp.float32))
                v_next = float(v_n[0, 0])

            mask_f = 0.0 if (terminated or truncated) else 1.0

            states     [t] = obs
            actions    [t] = int(action)
            rewards    [t] = float(rew)

            dones_term [t] = 1.0 if terminated else 0.0
            masks      [t] = mask_f

            values     [t] = float(v)
            next_values[t] = float(v_next)
            log_probs  [t] = float(logp)

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

        states_j      = jnp.asarray(states,      dtype=jnp.float32)
        actions_j     = jnp.asarray(actions,     dtype=jnp.int32  )
        rewards_j     = jnp.asarray(rewards,     dtype=jnp.float32)
        dones_term_j  = jnp.asarray(dones_term,  dtype=jnp.float32)
        masks_j       = jnp.asarray(masks,       dtype=jnp.float32)
        values_j      = jnp.asarray(values,      dtype=jnp.float32)
        next_values_j = jnp.asarray(next_values, dtype=jnp.float32)
        old_logp_j    = jnp.asarray(log_probs,   dtype=jnp.float32)

        advantages_j, returns_j = calculate_gae(rewards_j, values_j, next_values_j, dones_term_j, masks_j)
        advantages_j = (advantages_j - jnp.mean(advantages_j)) / (jnp.std(advantages_j) + 1e-8)

        last_total_loss  = 0.0
        last_actor_loss  = 0.0
        last_critic_loss = 0.0
        last_entropy     = 0.0
        last_kl          = 0.0
        last_clipfrac    = 0.0

        # PPO Update Loop, KL Early Stop
        stop_early = False
        for ep in range(epochs_per_update):
            rng, perm_rng = jax.random.split(rng)
            indices = jax.random.permutation(perm_rng, jnp.arange(steps_per_batch))

            for start in range(0, steps_per_batch, mini_batch_size):
                end    = start + mini_batch_size
                mb_idx = indices[start:end]

                actor_critic_params, opt_state, total_loss, actor_loss, critic_loss, entropy, approx_kl, clipfrac = train_step(
                    actor_critic_params,
                    opt_state,
                    states_j     [mb_idx],
                    actions_j    [mb_idx],
                    old_logp_j   [mb_idx],
                    advantages_j [mb_idx],
                    returns_j    [mb_idx],
                    values_j     [mb_idx]
                )

                last_total_loss  = float(total_loss )
                last_actor_loss  = float(actor_loss )
                last_critic_loss = float(critic_loss)
                last_entropy     = float(entropy    )
                last_kl          = float(approx_kl  )
                last_clipfrac    = float(clipfrac   )

                if last_kl > target_kl:
                    stop_early = True
                    break

            if stop_early:
                break

        avg_ep = float(np.mean(ep_returns)) if len(ep_returns) > 0 else float(ep_return)

        print(f"Update: {update:4d}, Steps: {global_steps:7d}, AvgEpRet: {avg_ep:8.1f}, "
              f"Loss: {last_total_loss:.4f}, Act: {last_actor_loss:.4f}, Cri: {last_critic_loss:.4f}, "
              f"Ent: {last_entropy:.4f}, KL: {last_kl:.4f}, ClipFr: {last_clipfrac:.3f}, EarlyStop: {stop_early}")

finally:
    env.close()
