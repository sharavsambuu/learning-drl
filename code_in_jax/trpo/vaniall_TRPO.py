import os
import random
import math
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np

debug_render    = False 
num_episodes    = 500
learning_rate   = 0.0005  # Slightly lower learning rate for TRPO
gamma           = 0.99
gae_lambda      = 0.97
policy_epochs   = 10
batch_size      = 64
mini_batch_size = 32
sync_steps      = 100
kl_target       = 0.01   # Target KL divergence for TRPO
max_backtracks  = 10     # Maximum backtracking steps for line search
backtrack_coeff = 0.8    # Backtracking coefficient for line search

class ActorNetwork(flax.nn.Module): # Same Actor network as PPO
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(flax.nn.Module): # Same Critic network as PPO
    def apply(self, x):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
        return output_dense_layer


env   = gym.make('CartPole-v1')
state = env.reset()
n_actions        = env.action_space.n

actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params)
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model     = flax.nn.Model(critic_module, critic_params)
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model)


@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def critic_inference(model, x):
    return model(x)

@jax.jit
def kl_divergence(p, q): # KL divergence between two distributions
    return jnp.sum(p * jnp.log(p / (q + 1e-8)), axis=-1) # Added small epsilon for stability

@jax.jit
def train_step(actor_optimizer, critic_optimizer, actor_model, critic_model, batch, old_actor_model_params):
    states, actions, old_log_probs, advantages, returns = batch

    def actor_loss_fn(actor_model):
        action_probabilities = actor_model(states)
        log_probs = jnp.log(action_probabilities[jnp.arange(len(actions)), actions])
        ratio = jnp.exp(log_probs - old_log_probs)
        surrogate_loss = -jnp.mean(ratio * advantages) # TRPO Surrogate objective (unclipped)

        kl_val = jnp.mean(kl_divergence(actor_model(states), flax.nn.Model(ActorNetwork.partial(n_actions=n_actions), old_actor_model_params)(states))) # KL divergence between new and old policy
        return surrogate_loss, kl_val

    def critic_loss_fn(critic_model): # Same Critic loss as PPO
        values = critic_model(states).reshape(-1)
        critic_loss = jnp.mean((values - returns)**2)
        return critic_loss

    # Calculate actor loss and gradients
    (actor_loss, kl_val), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_optimizer.target)

    # Compute critic loss and gradients (as in PPO)
    critic_grads, critic_loss = jax.value_and_grad(critic_loss_fn)(critic_optimizer.target)

    # Apply gradients to critic optimizer
    critic_optimizer = critic_optimizer.apply_gradient(critic_grads)

    # TRPO Policy Update using Conjugate Gradient and Line Search ---
    def HVP(params, vector, states, old_log_probs, advantages, damping=0.1): # Hessian-vector product
        def KL_grads(actor_params):
            log_probs = jnp.log(flax.nn.Model(ActorNetwork.partial(n_actions=n_actions), actor_params)(states)[jnp.arange(len(actions)), actions])
            kl_val = jnp.mean(kl_divergence(actor_model(states), flax.nn.Model(ActorNetwork.partial(n_actions=n_actions), actor_params)(states)))
            return kl_val
        return jax.grad(jax.grad(KL_grads), argnums=0)(params, vector) + damping * vector # Adding damping

    def conjugate_gradient(params, states, old_log_probs, advantages, max_iterations=10): # Conjugate gradient algorithm
        x = jnp.zeros_like(params)
        r = actor_grads # Use actor gradients as initial residual
        p = r
        rdotr = jnp.dot(r, r)
        for i in range(max_iterations):
            Avp       = HVP(params, p, states, old_log_probs, advantages)
            alpha     = rdotr / jnp.dot(p, Avp)
            x        += alpha * p
            r        -= alpha * Avp
            new_rdotr = jnp.dot(r, r)
            beta      = new_rdotr / rdotr
            p         = r + beta * p
            rdotr     = new_rdotr
        return x

    descent_direction = conjugate_gradient(actor_optimizer.target.params, states, old_log_probs, advantages) # Compute descent direction using CG
    descent_direction_flattened, _ = jax.tree_util.ravel_tree(descent_direction)
    descent_direction_step_size = jnp.sqrt(2 * kl_target / (jnp.dot(descent_direction_flattened, HVP(actor_optimizer.target.params, descent_direction, states, old_log_probs, advantages)))) # Calculate step size

    def full_step_actor_loss(step_size): # Loss function for line search
        actor_model_stepped_params = jax.tree_util.tree_map(lambda old_param, step: old_param - step_size * step, actor_optimizer.target.params, descent_direction)
        action_probabilities = flax.nn.Model(ActorNetwork.partial(n_actions=n_actions), actor_model_stepped_params)(states)
        log_probs = jnp.log(action_probabilities[jnp.arange(len(actions)), actions])
        ratio = jnp.exp(log_probs - old_log_probs)
        surrogate_loss = -jnp.mean(ratio * advantages)
        kl_val = jnp.mean(kl_divergence(actor_model(states), flax.nn.Model(ActorNetwork.partial(n_actions=n_actions), actor_model_stepped_params)(states)))
        return surrogate_loss, kl_val

    step_size = 1.0 # Initial step size for line search
    for i in range(max_backtracks): # Line search loop
        actor_loss_new, kl_val_new = full_step_actor_loss(step_size)
        if actor_loss_new > actor_loss and kl_val_new <= kl_target: # Accept step if loss improves and KL constraint is satisfied
            break
        step_size *= backtrack_coeff # Backtrack step size
    else:
        step_size = 0.0 # If line search fails, set step size to 0

    actor_model_params_updated = jax.tree_util.tree_map(lambda old_param, step: old_param - step_size * step, actor_optimizer.target.params, descent_direction) # Apply step size to update actor parameters
    actor_optimizer = actor_optimizer.replace(target=actor_model.replace(params=actor_model_params_updated)) # Update actor optimizer with new parameters

    return actor_optimizer, critic_optimizer, actor_loss, critic_loss, kl_val # Return KL value for monitoring


global_steps = 0
try:
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        episode_states, episode_actions, episode_rewards_list, episode_log_probs, episode_values = [], [], [], [], []
        while True:
            global_steps += 1
            action_probabilities = actor_inference(actor_optimizer.target, jnp.asarray([state]))
            action_probabilities = np.array(action_probabilities[0])
            action = np.random.choice(n_actions, p=action_probabilities)
            value = critic_inference(critic_optimizer.target, jnp.asarray([state]))

            log_prob = jnp.log(action_probabilities[action])

            new_state, reward, done, _ = env.step(int(action))

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_log_probs.append(log_prob)
            episode_values.append(value[0])

            state = new_state
            episode_rewards.append(reward)

            if debug_render:
                env.render()

            if done:
                print(f"{episode} episode, reward: {sum(episode_rewards)}")
                break

        # Calculate GAE advantages and returns (same as PPO)
        values_np = np.array(episode_values + [0])
        advantages = np.zeros_like(episode_rewards_list, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(episode_rewards_list))):
            delta = episode_rewards_list[t] + gamma * values_np[t + 1] - values_np[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam

        returns = advantages + np.array(episode_values)

        # Prepare batch data (same as PPO)
        batch_data = (
            np.array(episode_states),
            np.array(episode_actions),
            np.array(episode_log_probs),
            advantages,
            returns
        )

        old_actor_model_params_for_kl = actor_optimizer.target.params # Store old policy parameters for KL constraint

        # Train TRPO for multiple epochs with mini-batches (same as PPO, but with TRPO train_step)
        for _ in range(policy_epochs):
            perm = np.random.permutation(len(episode_states))
            for start_idx in range(0, len(episode_states), mini_batch_size):
                mini_batch_idx = perm[start_idx:start_idx + mini_batch_size]
                mini_batch = tuple(arr[mini_batch_idx] for arr in batch_data)
                actor_optimizer, critic_optimizer, actor_loss, critic_loss, kl_val = train_step(
                    actor_optimizer, critic_optimizer, actor_model, critic_model, mini_batch, old_actor_model_params_for_kl
                )
                print(f"KL Divergence: {kl_val:.4f}", end='\r') # Monitor KL divergence during training

finally:
    env.close()