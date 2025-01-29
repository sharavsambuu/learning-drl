import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import jax.tree_util


num_episodes           = 500
learning_rate          = 0.003  
gamma                  = 0.99
kl_target              = 0.01
batch_size             = 128  
n_actions              = 2

# Epsilon-greedy exploration
epsilon_start          = 0.2  
epsilon_end            = 0.01
epsilon_decay_episodes = num_episodes / 4  

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


env             = gym.make('CartPole-v1', render_mode='human')
state_dim       = env.observation_space.shape[0]
policy_module   = PolicyNetwork(n_actions=n_actions)
value_module    = ValueNetwork()
dummy_input     = jnp.zeros((1, state_dim))
policy_params   = policy_module.init(jax.random.PRNGKey(0), dummy_input)['params']
value_params    = value_module .init(jax.random.PRNGKey(1), dummy_input)['params']
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

    for _ in range(5):  # Increased value update iterations
        loss, grads = jax.value_and_grad(value_loss_fn)(value_params)
        updates, new_value_opt_state = value_optimizer.update(grads, value_opt_state, value_params)
        value_params = optax.apply_updates(value_params, updates)
    return value_params, new_value_opt_state, loss

@jax.jit
def calculate_kl_divergence(old_params, new_params, states):
    old_probs = policy_module.apply({'params': old_params}, states)
    new_probs = policy_module.apply({'params': new_params}, states)
    kl = jnp.sum(old_probs * (jnp.log(old_probs + 1e-6) - jnp.log(new_probs + 1e-6)), axis=-1)
    return jnp.mean(kl)

@jax.jit
def compute_surrogate_loss(new_params, old_params, states, actions, advantages):
    new_probs = policy_module.apply({'params': new_params}, states)
    old_probs = policy_module.apply({'params': old_params}, states)
    new_probs_for_actions = gather(new_probs, actions)
    old_probs_for_actions = gather(old_probs, actions)
    ratio = new_probs_for_actions / (old_probs_for_actions + 1e-6)
    surrogate_loss = ratio * advantages
    return jnp.mean(surrogate_loss)

def conjugate_gradient(old_params, states, actions, advantages, b, num_iterations=15): # Increased CG iterations
    x = jnp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = jnp.dot(r, r)
    for _ in range(num_iterations):
        def hvp_loss(params):
            probs = policy_module.apply({'params': params}, states)
            probs_for_actions = gather(probs, actions)
            ratio = probs_for_actions / (gather(policy_module.apply({'params': old_params}, states), actions) + 1e-6)
            surrogate_loss = ratio * advantages
            return jnp.mean(surrogate_loss)
        Ap = flatten_params(hvp(hvp_loss, (old_params,), (unflatten_params(p, old_params),)))
        alpha = rdotr / jnp.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = jnp.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def linesearch(old_params, new_params, states, actions, advantages, expected_improvement, max_backtracks=15, backtrack_ratio=0.8): # Increased backtracks
    for step in range(max_backtracks):
        ratio = backtrack_ratio ** step
        step_params = jax.tree.map(lambda p, u: p + ratio * u, old_params, unflatten_params(new_params, old_params))
        loss = compute_surrogate_loss(step_params, old_params, states, actions, advantages)
        improvement = loss - compute_surrogate_loss(old_params, old_params, states, actions, advantages)
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

def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]

try:
    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], []
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay_episodes)

        while not done:
            action_probs = policy_inference(policy_params, jnp.array([state]))

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(jnp.argmax(action_probs, axis=-1).item())

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        dones = jnp.array(dones)

        advantages = calculate_gae(value_params, states, rewards, dones)
        returns = advantages + jnp.squeeze(value_inference(value_params, states), axis=-1)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-6)

        value_params, value_opt_state, value_loss = value_update(value_params, value_opt_state, states, returns)

        old_params = policy_params
        b = flatten_params(jax.grad(compute_surrogate_loss)(old_params, old_params, states, actions, advantages))
        search_direction = conjugate_gradient(old_params, states, actions, advantages, b)

        expected_improvement = compute_surrogate_loss(old_params, old_params, states, actions, advantages) + 1e-6

        # Update policy parameters
        policy_params = linesearch(old_params, search_direction, states, actions, advantages, expected_improvement)

        # Calculate KL divergence after the update
        kl_divergence = calculate_kl_divergence(old_params, policy_params, states)
        if kl_divergence > 1.5 * kl_target:
            print(f'Episode {episode}: Early stopping due to high KL divergence: {kl_divergence}')

        print(f"Episode: {episode}, Total Reward: {total_reward}, Value Loss: {value_loss:.4f}, KL Divergence: {kl_divergence:.4f}, Epsilon: {epsilon:.4f}")

finally:
    env.close()