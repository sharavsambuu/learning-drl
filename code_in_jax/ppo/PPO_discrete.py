#
# PPO (Discrete) + Orthogonal Init + Gradient Clipping + GAE(λ) + Proper Truncation Handling
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
learning_rate       = 3e-4    # 0.0003

gamma               = 0.99
gae_lambda          = 0.95

clip_epsilon        = 0.2
entropy_coeff       = 0.001
value_coeff         = 0.5
max_grad_norm       = 0.5     # Critical for PPO stability

epochs_per_update   = 10
mini_batch_size     = 256

debug_render        = True
render_update_every = 10
render_sleep        = 0.0


class ActorCriticNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        # Orthogonal Initialization prevents vanishing/exploding gradients early on
        init_ortho = nn.initializers.orthogonal(np.sqrt(2))
        init_layer = nn.initializers.orthogonal(0.01) # Small scale for policy
        init_value = nn.initializers.orthogonal(1.0)

        # Shared backbone
        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64, kernel_init=init_ortho)(x)
        x = nn.tanh(x)

        # Heads
        logits = nn.Dense(features=self.n_actions, kernel_init=init_layer)(x)
        value  = nn.Dense(features=1,              kernel_init=init_value)(x)

        return logits, value


env = gym.make("CartPole-v1", render_mode="human" if debug_render else None)
state_dim   = env.observation_space.shape[0]
n_actions   = env.action_space.n

actor_critic_module = ActorCriticNetwork(n_actions=n_actions)

rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng, 2)

dummy_input         = jnp.zeros((1, state_dim), dtype=jnp.float32)
actor_critic_params = actor_critic_module.init(init_rng, dummy_input)["params"]

# Chain: Global Norm Clipping -> Adam
optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(learning_rate, eps=1e-5)
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
def calculate_gae(rewards, values, last_val, dones):
    # Construct full value sequence: [v0, v1, ..., vT, v_final]
    # dones indicates if v_next should be masked (0 if terminated, 1 if alive/truncated)
    
    # We shift values to get next_values without re-inferencing everything
    next_values = jnp.concatenate([values[1:], last_val.reshape(1)])
    
    # Standard GAE
    # If done=1 (terminated), we do not look at next_values
    deltas = rewards + gamma * (1.0 - dones) * next_values - values

    def scan_fn(carry, x):
        delta, done = x
        carry = delta + gamma * gae_lambda * (1.0 - done) * carry
        return carry, carry

    _, adv = jax.lax.scan(
        scan_fn,
        jnp.array(0.0, dtype=deltas.dtype),
        (deltas, dones),
        reverse=True
    )
    returns = adv + values
    return adv, returns

@jax.jit
def ppo_loss(params, states, actions, old_log_probs, advantages, returns):
    logits, values = ac_inference(params, states)
    values = values.squeeze(-1)

    new_log_probs = logprob_from_logits(logits, actions)
    ratio         = jnp.exp(new_log_probs - old_log_probs)

    pg_loss1   = -advantages * ratio
    pg_loss2   = -advantages * jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value clipped loss (optional but often standard, sticking to simple MSE here is fine too)
    critic_loss = jnp.square(returns - values).mean()
    
    entropy     = entropy_from_logits(logits)
    total_loss  = actor_loss + value_coeff * critic_loss - entropy_coeff * entropy
    
    return total_loss, (actor_loss, critic_loss, entropy)

@jax.jit
def train_step(params, opt_state, states, actions, old_log_probs, advantages, returns):
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (loss, (actor_loss, critic_loss, entropy)), grads = grad_fn(
        params, states, actions, old_log_probs, advantages, returns
    )
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, actor_loss, critic_loss, entropy


updates      = int(total_timesteps // steps_per_batch)
global_steps = 0

try:
    for update in range(updates):
        # Buffer initialization
        states      = np.zeros((steps_per_batch, state_dim), dtype=np.float32)
        actions     = np.zeros((steps_per_batch,), dtype=np.int32)
        rewards     = np.zeros((steps_per_batch,), dtype=np.float32)
        dones       = np.zeros((steps_per_batch,), dtype=np.float32)
        values      = np.zeros((steps_per_batch,), dtype=np.float32)
        log_probs   = np.zeros((steps_per_batch,), dtype=np.float32)

        ep_returns = []
        ep_return  = 0.0

        obs, info = env.reset()
        obs = np.array(obs, dtype=np.float32)

        do_render = debug_render and (update % render_update_every == 0)

        for t in range(steps_per_batch):
            rng, a_rng = jax.random.split(rng)

            # Inference current state ONLY
            logits, v = ac_inference(actor_critic_params, jnp.array([obs], dtype=jnp.float32))
            logits = logits[0]
            v      = v[0, 0]

            action = jax.random.categorical(a_rng, logits).astype(jnp.int32)
            logp   = logprob_from_logits(logits.reshape(1, -1), action.reshape(1,))

            # Step
            nobs, rew, terminated, truncated, info = env.step(int(action))
            
            # Logic: If terminated (died), value next is 0. 
            # If truncated (time limit), value next is V(s'). 
            # We record 'terminated' as the done flag for GAE.
            done_bool = terminated
            
            states    [t] = obs
            actions   [t] = int(action)
            rewards   [t] = float(rew)
            dones     [t] = 1.0 if done_bool else 0.0
            values    [t] = float(v)
            log_probs [t] = float(logp[0])

            ep_return += float(rew)
            global_steps += 1
            
            nobs = np.array(nobs, dtype=np.float32)

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

        # Bootstrapping: Value of the very last state
        # If the last step was a reset, obs is fresh start (v approx 0 or baseline)
        # If the last step was valid, obs is s_{T+1}, we need v(s_{T+1})
        _, last_val_j = ac_inference(actor_critic_params, jnp.array([obs], dtype=jnp.float32))
        last_val_j = last_val_j[0, 0]

        # Convert to JAX
        states_j      = jnp.asarray(states,    dtype=jnp.float32)
        actions_j     = jnp.asarray(actions,   dtype=jnp.int32  )
        rewards_j     = jnp.asarray(rewards,   dtype=jnp.float32)
        dones_j       = jnp.asarray(dones,     dtype=jnp.float32)
        values_j      = jnp.asarray(values,    dtype=jnp.float32)
        old_logp_j    = jnp.asarray(log_probs, dtype=jnp.float32)

        # GAE
        advantages_j, returns_j = calculate_gae(rewards_j, values_j, last_val_j, dones_j)
        
        # Normalize Advantages
        advantages_j = (advantages_j - jnp.mean(advantages_j)) / (jnp.std(advantages_j) + 1e-8)

        # PPO Update Loop
        for ep in range(epochs_per_update):
            rng, perm_rng = jax.random.split(rng)
            indices = jax.random.permutation(perm_rng, jnp.arange(steps_per_batch))

            for start in range(0, steps_per_batch, mini_batch_size):
                end    = start + mini_batch_size
                mb_idx = indices[start:end]

                actor_critic_params, opt_state, total_loss, actor_loss, critic_loss, entropy = train_step(
                    actor_critic_params,
                    opt_state,
                    states_j     [mb_idx],
                    actions_j    [mb_idx],
                    old_logp_j   [mb_idx],
                    advantages_j [mb_idx],
                    returns_j    [mb_idx]
                )

        avg_ep = float(np.mean(ep_returns)) if len(ep_returns) > 0 else float(ep_return)

        print(f"Update: {update:4d}, Steps: {global_steps:7d}, AvgEpRet: {avg_ep:8.1f}, "
              f"Loss: {float(total_loss):.4f}, Act: {float(actor_loss):.4f}, Cri: {float(critic_loss):.4f}, Ent: {float(entropy):.4f}")

finally:
    env.close()