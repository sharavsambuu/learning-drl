import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import random


num_episodes      = 1000
learning_rate     = 0.0005  # Learning rate for PPO, often smaller than TRPO
gamma             = 0.99
clip_epsilon      = 0.2     # Clipping parameter for PPO's objective
entropy_coeff     = 0.01    # Entropy coefficient, encourage exploration
value_coeff       = 0.5     # Value loss coefficient
batch_size        = 64
epochs_per_update = 4       # Number of epochs to train on collected data
gae_lambda        = 0.95    # Lambda for GAE, between 0 and 1
mini_batch_size   = 32


class ActorCriticNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        # Common layers for both actor and critic
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        # Actor head
        actor_output  = nn.Dense(features=self.n_actions)(x)
        actor_probs   = nn.softmax(actor_output)

        # Critic head
        critic_output = nn.Dense(features=1)(x)

        return actor_probs, critic_output


env         = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
n_actions   = env.action_space.n

actor_critic_module = ActorCriticNetwork(n_actions=n_actions)
dummy_input         = jnp.zeros(state.shape)
actor_critic_params = actor_critic_module.init(jax.random.PRNGKey(0), dummy_input)['params']
optimizer           = optax.adam(learning_rate)
opt_state           = optimizer.init(actor_critic_params)


@jax.jit
def policy_inference(params, x):
    action_probs, _ = actor_critic_module.apply({'params': params}, x)
    return action_probs

@jax.jit
def get_values(params, x):
    _, values = actor_critic_module.apply({'params': params}, x)
    return values

@jax.jit
def calculate_gae(rewards, values, next_values, dones):
    advantages = []
    advantage = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        advantage = delta + gamma * gae_lambda * (1 - dones[i]) * advantage
        advantages.insert(0, advantage)
    return jnp.array(advantages)

@jax.jit
def ppo_update(params, opt_state, states, actions, old_log_probs, advantages, returns):
    def loss_fn(params, states, actions, old_log_probs, advantages, returns):
        action_probs, values = actor_critic_module.apply({'params': params}, states)
        values = values.squeeze()

        # Actor loss
        log_probs = jnp.log(gather(action_probs, actions) + 1e-6)
        ratio = jnp.exp(log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Critic loss
        critic_loss = jnp.square(returns - values).mean()

        # Entropy loss
        entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-6), axis=-1).mean()

        # Total loss
        total_loss = actor_loss + value_coeff * critic_loss - entropy_coeff * entropy

        return total_loss, (actor_loss, critic_loss, entropy)

    for _ in range(epochs_per_update):
        num_samples = len(states)
        indices = jnp.arange(num_samples)
        indices = jax.random.permutation(jax.random.PRNGKey(0), indices)
        
        for start in range(0, num_samples, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]
            
            mini_batch_states       = states      [batch_indices]
            mini_batch_actions      = actions     [batch_indices]
            mini_batch_old_log_probs = old_log_probs[batch_indices]
            mini_batch_advantages   = advantages  [batch_indices]
            mini_batch_returns      = returns     [batch_indices]
            
            (total_loss, (actor_loss, critic_loss, entropy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, mini_batch_states, mini_batch_actions, mini_batch_old_log_probs, mini_batch_advantages, mini_batch_returns
            )
            
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            params                 = optax.apply_updates(params, updates)

    return params, new_opt_state, total_loss, actor_loss, critic_loss, entropy

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]


master_rng = jax.random.PRNGKey(0)
try:
    for episode in range(num_episodes):
        state, info  = env.reset()
        state        = np.array(state, dtype=np.float32)
        done         = False
        total_reward = 0
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        master_rng, loop_rng = jax.random.split(master_rng)

        while not done:
            loop_rng, action_rng = jax.random.split(loop_rng)
            
            action_probs = policy_inference(actor_critic_params, jnp.array([state]))
            value        = get_values(actor_critic_params, jnp.array([state]))
            action       = jax.random.categorical(action_rng, action_probs[0])
            log_prob     = jnp.log(action_probs[0][action] + 1e-6)

            next_state, reward, terminated, truncated, info = env.step(int(action))
            done          = terminated or truncated
            next_state    = np.array(next_state, dtype=np.float32)
            total_reward += reward

            states   .append(state)
            actions  .append(int(action))
            rewards  .append(reward)
            dones    .append(done)
            log_probs.append(log_prob)
            values   .append(value[0])

            state = next_state

        next_value = get_values(actor_critic_params, jnp.array([state]))[0]
        advantages = calculate_gae(jnp.array(rewards), jnp.array(values), next_value, jnp.array(dones))
        returns    = advantages + jnp.array(values)

        actor_critic_params, opt_state, total_loss, actor_loss, critic_loss, entropy = ppo_update(
            actor_critic_params,
            opt_state,
            jnp.array(states),
            jnp.array(actions),
            jnp.array(log_probs),
            advantages,
            returns
        )

        print(f"Episode: {episode}, Total Reward: {total_reward}, Total Loss: {total_loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")

finally:
    env.close()