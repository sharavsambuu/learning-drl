import gym
from gym import spaces
import numpy as np
import random
import flax
import jax
from jax import numpy as jnp

# Minimal Custom Cooperative Grid World Environment 
class CooperativeGridWorldEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(CooperativeGridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.n_agents   = 2 # Two agents
        self.action_space = spaces.Discrete(4)  # Discrete actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Tuple([ # Individual observation space for each agent (just their own position)
            spaces.Discrete(grid_size * grid_size),
            spaces.Discrete(grid_size * grid_size)
        ])
        self.agents_pos = None # Agent positions

    def reset(self):
        self.agents_pos = [ # Random initial positions for both agents
            (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)),
            (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        ]
        return self._get_obs()

    def _get_obs(self): # Simple observation: flattened index of agent's position
        return tuple(agent_pos[0] * self.grid_size + agent_pos[1] for agent_pos in self.agents_pos)

    def step(self, actions):
        new_agents_pos = list(self.agents_pos)
        for agent_id, action in enumerate(actions):
            row, col = self.agents_pos[agent_id]
            if action == 0: row = max(0, row - 1)                    # Up
            elif action == 1: row = min(self.grid_size - 1, row + 1) # Down
            elif action == 2: col = max(0, col - 1)                  # Left
            elif action == 3: col = min(self.grid_size - 1, col + 1) # Right
            new_agents_pos[agent_id] = (row, col)
        self.agents_pos = new_agents_pos

        reward = 0 # Cooperative reward: Agents get reward only when they are at the *same* location
        if self.agents_pos[0] == self.agents_pos[1]:
            reward = 1
        done = False
        truncated = False # Add truncated for gym 0.26 compatibility
        info = {}
        return self._get_obs(), reward, done, truncated, info

    def render(self, mode='human'): # Simple text-based rendering
        grid = [['-' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for agent_id, agent_pos in enumerate(self.agents_pos):
            row, col = agent_pos
            grid[row][col] = str(agent_id + 1) # Mark agents as '1' and '2'
        for row in grid:
            print(' '.join(row))


# Independent Q-Learning Agents (Minimal DQNs) 
class SimpleQNetwork(flax.nn.Module): # Very simple Q-Network
    def apply(self, x, n_actions):
        dense_layer      = flax.nn.Dense(x, features=16) # Smaller network
        activation_layer = flax.nn.relu(dense_layer)
        output_layer   = flax.nn.Dense(activation_layer, n_actions)
        return output_layer

n_actions = 4 # Up, Down, Left, Right
n_agents  = 2 # Two agents

q_networks        = []
target_q_networks = []
optimizers        = []

for agent_id in range(n_agents): # Create separate networks and optimizers for each agent
    q_module          = SimpleQNetwork.partial(n_actions=n_actions)
    _, params       = q_module.init_by_shape(jax.random.PRNGKey(0), [()]) # Scalar input for position index
    q_network       = flax.nn.Model(q_module, params)
    target_q_network= flax.nn.Model(q_module, params)
    optimizer       = flax.optim.Adam(learning_rate).create(q_network)
    q_networks.append(q_network)
    target_q_networks.append(target_q_network)
    optimizers.append(optimizer)


@jax.jit # Policy function for each agent (Independent Q-Learning - each agent acts greedily based on its own Q-network)
def policy(model, x):
    predicted_q_values = model(x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action

@jax.vmap # vmap for Q-learning loss (same as single-agent DQN)
def q_learning_loss(q_value_vec, target_q_value_vec, action, reward, done):
    td_target = reward + gamma*jnp.max(target_q_value_vec)*(1.-done)
    td_error  = jax.lax.stop_gradient(td_target) - q_value_vec[action]
    return jnp.square(td_error)

@jax.jit # Train step for a *single* agent's Q-network
def train_step(optimizer, target_model, batch):
    def loss_fn(model):
        predicted_q_values      = model(batch[0])
        target_q_values         = target_model(batch[3])
        return jnp.mean(
                q_learning_loss(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

num_episodes = 3000 # Increased episodes for simpler environment

try:
    for episode in range(num_episodes):
        state = env.reset() # Reset environment
        episode_rewards = 0
        done = False
        while not done:
            actions = []
            for agent_id in range(n_agents): # Each agent chooses action independently
                agent_state = jnp.array(state[agent_id]) # Get individual agent state
                action = policy(optimizers[agent_id].target, agent_state)[0] # Independent policy for each agent
                actions.append(action)

            next_state, reward, done, truncated, info = env.step(actions) # Environment step with joint actions

            # Store experience (per agent, but using joint reward) -  For simplicity, using shared replay for both agents
            for agent_id in range(n_agents):
                agent_state = jnp.array(state[agent_id])
                agent_next_state = jnp.array(next_state[agent_id])
                experience = (agent_state, actions[agent_id], reward, agent_next_state, float(done)) # Store experience for *each* agent
                per_memory.add(0, experience) # Initial priority 0 - You can implement proper PER later


            # Train agents (Independent Q-Learning - each agent's network is trained independently)
            if len(per_memory) > batch_size:
                batch = per_memory.sample(batch_size) # Sample batch from *shared* replay buffer
                # Separate batches for each agent (but sampled from shared memory for simplicity)
                batches = [ [batch[i][1][j] for i in range(batch_size)] for j in range(5)] # Reorganize batch data

                for agent_id in range(n_agents): # Train each agent's network independently
                    agent_batch = (jnp.asarray(batches[0]), jnp.asarray(batches[1]), jnp.asarray(batches[2]), jnp.asarray(batches[3]), jnp.asarray(batches[4])) # Create batch for *this* agent
                    optimizers[agent_id], loss = train_step(optimizers[agent_id], target_q_networks[agent_id], agent_batch) # Train agent's Q-network


            state = next_state
            episode_rewards += reward # Sum of joint rewards

            if debug_render:
                env.render()
                time.sleep(0.1)

        if episode % sync_steps == 0: # Target network sync (Independent Target Networks for each agent)
            for agent_id in range(n_agents):
                target_q_networks[agent_id] = target_q_networks[agent_id].replace(params=optimizers[agent_id].target.params)

        print(f"Episode {episode}, Reward: {episode_rewards}")

finally:
    env.close()