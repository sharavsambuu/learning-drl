import os
import random
import math
import time
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np
import pybullet_envs


debug_render  = True
num_episodes  = 500
batch_size    = 128
learning_rate = 0.001
sync_steps    = 1
memory_length = 10000

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99  # discount factor
alpha         = 0.6   # entropy tradeoff factor
tau           = 0.005 # soft update

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2*capacity - 1)
        self.data     = np.zeros(capacity, dtype=object)
    def _propagate(self, idx, change):
        parent             = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])
    def total(self):
        return self.tree[0]
    def add(self, p, data):
        idx                   = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write           += 1
        if self.write >= self.capacity:
            self.write = 0
    def update(self, idx, p):
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx     = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PERMemory:
    e = 0.01
    a = 0.6
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    def _get_priority(self, error):
        return (error+self.e)**self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample) 
    def sample(self, n):
        batch   = []
        segment = self.tree.total()/n
        for i in range(n):
            a = segment*i
            b = segment*(i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)



class CommonNetwork(flax.nn.Module):
    def apply(self, x, n_outputs):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_outputs)
        return output_layer


class TwinQNetwork(flax.nn.Module):
    def apply(self, x):
        q1 = CommonNetwork(x, 1)
        q2 = CommonNetwork(x, 1)
        return q1, q2


# https://github.com/google/jax/issues/2173
# https://github.com/pytorch/pytorch/blob/master/torch/distributions/normal.py#L65
@jax.jit
def gaussian_normal(key, mu, sigma):
    normals = mu + jax.random.normal(key, mu.shape)*sigma
    return normals

#https://github.com/pytorch/pytorch/blob/master/torch/distributions/normal.py#L70
@jax.jit
def log_prob(mean, scale, value):
    var       = (jnp.square(scale))
    log_scale = jnp.log(scale)
    return -((value - mean) ** 2) / (2 * var) - log_scale - jnp.log(jnp.sqrt(2 * jnp.pi))

class GaussianPolicy(flax.nn.Module):
    def apply(self, x, n_actions, key=None, sample=False, clip_min=-1., clip_max=1.):
        policy_layer  = CommonNetwork(x, n_actions*2)
        mean, log_std = jnp.split(policy_layer, 2, axis=-1)
        log_std       = jnp.clip(log_std, clip_min, clip_max)
        if sample:
            stds      = jnp.exp(log_std)
            xs        = gaussian_normal(key, mean, stds)
            actions   = flax.nn.tanh(xs)
            log_probs = log_prob(mean, stds, xs) - jnp.log(1-jnp.square(actions)+1e-6)
            entropies = -jnp.sum(log_probs, axis=1, keepdims=True)
            return actions, entropies, flax.nn.tanh(mean)
        else:
            return mean, log_std


# https://github.com/henry-prior/jax-rl/blob/436b009cd97475b75be3b192a0ba761152950f41/models.py#L98
class Constant(flax.nn.Module):
    def apply(self, start_value, dtype=jnp.float32):
        value = self.param('value', (1,), flax.nn.initializers.ones)
        return start_value * jnp.asarray(value, dtype)


#environment_name = 'HumanoidFlagrunHarderBulletEnv-v0' 
environment_name = 'LunarLanderContinuous-v2' 
env              = gym.make(environment_name)

if debug_render:
    env.render(mode="human")
state = env.reset()


clip_min=min(np.array([env.action_space.high, env.action_space.low]).flatten())
clip_max=max(np.array([env.action_space.high, env.action_space.low]).flatten())


critic_module = TwinQNetwork.partial()
_, params     = critic_module.init_by_shape(
    jax.random.PRNGKey(0), 
    [(env.observation_space.shape[0]+env.action_space.shape[0],)])

critic        = flax.nn.Model(critic_module, params)
target_critic = flax.nn.Model(critic_module, params)

actor_module    = GaussianPolicy.partial(
    n_actions   = env.action_space.shape[0], 
    key         = jax.random.PRNGKey(0),
    clip_min    = clip_min,
    clip_max    = clip_max
    )
_, actor_params = actor_module.init_by_shape(
    jax.random.PRNGKey(0),
    [env.observation_space.shape])
actor           = flax.nn.Model(actor_module, actor_params)

constant_module    = Constant.partial(start_value=alpha)
_, constant_params = constant_module.init(jax.random.PRNGKey(0))
constant_model     = flax.nn.Model(constant_module, constant_params)


critic_optimizer = flax.optim.Adam(learning_rate).create(critic)
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor)
alpha_optimizer  = flax.optim.Adam(learning_rate).create(constant_model)



@jax.jit
def actor_inference(model, state, key):
    actions, entropies, tanh_means = model(state, key=key, sample=True)
    return actions, entropies, tanh_means

@jax.jit
def critic_inference(model, state, action):
    state_action = jnp.concatenate((state, action), axis=-1)
    q1, q2       = model(state_action)
    return q1, q2

@jax.jit
def target_critic_inference(actor_model, target_critic_model, next_state, reward, done, alpha, key):
    next_actions, next_entropies, _ = actor_model(next_state, key=key, sample=True)
    next_state_action               = jnp.concatenate((next_state, next_actions), axis=-1)
    next_q1, next_q2                = target_critic_model(next_state_action)
    min_q                           = jnp.min([next_q1, next_q2], axis=0)
    next_q                          = min_q+alpha*next_entropies
    rewards                         = jnp.reshape(reward, (reward.shape[0], 1))
    dones                           = jnp.subtract(1.0, done)
    dones                           = jnp.reshape(dones, (dones.shape[0], 1))
    target_q                        = rewards+gamma*next_q*dones
    return target_q


# https://github.com/henry-prior/jax-rl/blob/436b009cd97475b75be3b192a0ba761152950f41/utils.py#L52
@jax.vmap
def double_mse(q1, q2, qt):
    return jnp.square(qt-q1).mean() + jnp.square(qt-q2).mean()

@jax.jit
def backpropagate_critic(
    critic_optimizer, actor_optimizer,
    target_critic,
    batch,
    alpha,
    key
    ):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    # batch[5] - weights

    target_q = jax.lax.stop_gradient(
        target_critic_inference(
            actor_optimizer.target, 
            target_critic, 
            batch[3],
            batch[2],
            batch[4], 
            alpha, 
            key
        )
    )

    def loss_fn(critic_model):
        state_action = jnp.concatenate((batch[0], batch[1]), axis=-1)
        q1, q2       = critic_model(state_action)
        critic_loss  = double_mse(q1, q2, target_q)
        return jnp.mean(critic_loss)
    loss, gradients  = jax.value_and_grad(loss_fn)(critic_optimizer.target)
    critic_optimizer = critic_optimizer.apply_gradient(gradients) 

    q1, _ = jax.lax.stop_gradient(
        critic_inference(
            critic_optimizer.target, 
            batch[0],
            batch[1]
        )
    )    
    td_errors = jnp.abs(q1-target_q)
    
    return critic_optimizer, td_errors


@jax.jit
def backpropagate_actor(optimizer, critic, batch, alpha, key):
    # batch[0] - states
    # batch[1] - weights
    def loss_fn(actor_model):
        sampled_actions, entropies, _ = actor_inference(
            optimizer.target, 
            batch[0], 
            key
            )
        q1, q2 = critic_inference(
            critic, 
            batch[0],
            sampled_actions
            )
        q_values   = jnp.min([q1, q2], axis=0)
        actor_loss = jnp.mean(
            jnp.multiply(
                (-q_values - entropies*alpha),
                batch[1]
            )
        )
        return actor_loss

    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)

    return optimizer, loss

# https://github.com/henry-prior/jax-rl/blob/436b009cd97475b75be3b192a0ba761152950f41/utils.py#L43
@jax.jit
def soft_update(model, target_model, tau):
    update_params = jax.tree_multimap(
        lambda m1, mt: tau*m1 + (1-tau)*mt,
        model.params, target_model.params
        )
    return target_model.replace(params=update_params)





per_memory   = PERMemory(memory_length)
rng          = jax.random.PRNGKey(0)

global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                rng, new_key  = jax.random.split(rng)
                actions, _, _ = actor_inference(
                    actor, 
                    jnp.asarray([state]), 
                    new_key)
                action = actions[0]

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)

            next_state, reward, done, _ = env.step(action)

            # sample цуглуулж жинг тооцон PER санах ойруу хийх
            q1, q2 = critic_inference(
                critic, 
                jnp.asarray([state]),
                jnp.asarray([action])
                )
            rng, new_key = jax.random.split(rng)
            target_q     = jax.lax.stop_gradient(
                target_critic_inference(
                    actor, 
                    target_critic, 
                    jnp.asarray([next_state]), 
                    jnp.asarray([reward]), 
                    jnp.asarray([int(done)]), 
                    alpha, 
                    new_key
                )
            )
            td_error = jnp.abs(q1-target_q)
            td_error = td_error[0][0]
            per_memory.add(td_error, (state, action, reward, next_state, int(done)))

            # сургах batch бэлтгэх
            batch = per_memory.sample(batch_size)
            states, actions, rewards, next_states, dones, weights = [], [], [], [], [], []
            for i in range(batch_size):
                try:
                    states.append     (batch[i][1][0])
                    actions.append    (batch[i][1][1])
                    rewards.append    (batch[i][1][2])
                    next_states.append(batch[i][1][3])
                    dones.append      (batch[i][1][4])
                    weights.append    (batch[i][0])
                except Exception as e:
                    continue
            
            # critic неорон сүлжээг сургах
            rng, new_key = jax.random.split(rng)
            critic_optimizer, td_errors = backpropagate_critic(
                critic_optimizer, actor_optimizer,
                target_critic,
                (
                    jnp.asarray(states),
                    jnp.asarray(actions),
                    jnp.asarray(rewards),
                    jnp.asarray(next_states),
                    jnp.asarray(dones),
                    jnp.asarray(weights)
                ),
                alpha,
                new_key
                )
            
            # PER санах ойны жинг шинэчлэх
            new_td_errors = np.array(td_errors).flatten()
            for i in range(batch_size):
                idx = batch[i][0]
                per_memory.update(idx, new_td_errors[i])

            # actor неорон сүлжээг сургах
            rng, new_key = jax.random.split(rng)
            actor_optimizer, actor_loss = backpropagate_actor(
                actor_optimizer,
                critic_optimizer.target,
                (
                    jnp.asarray(states),
                    jnp.asarray(weights)
                ), 
                alpha, 
                new_key
                )

            # сайжирсэн жингүүдээр target_critic неорон сүлжээг шинэчлэх
            if global_steps%sync_steps==0:
                target_critic = soft_update(critic_optimizer.target, target_critic, tau)


            episode_rewards.append(reward)
            state = next_state

            if debug_render:
                #time.sleep(1. / 60)
                env.render(mode="human")

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
