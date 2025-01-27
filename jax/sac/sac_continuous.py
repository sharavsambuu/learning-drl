import os
import random
import math
import time
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np
#import pybullet_envs 

debug_render  = False                                                                          # render-ийг debug хийх үед True болгоно уу
num_episodes  = 800                                                                            # episode-ийн тоог нэмэгдүүлэх
batch_size    = 256                                                                            # batch size-ийг нэмэгдүүлэх
learning_rate = 0.0003                                                                         # learning rate-ийг багасгах нь сургалтыг тогтворжуулна
sync_steps    = 1
memory_length = 10000

epsilon       = 1.0
epsilon_decay = 0.0005                                                                         # epsilon decay-ийг багасгах
epsilon_max   = 1.0
epsilon_min   = 0.1                                                                            # epsilon min-ийг нэмэгдүүлэх

gamma         = 0.99                                                                           # discount factor
alpha         = 0.2                                                                            # entropy tradeoff factor - багасгах
tau           = 0.005                                                                          # soft update

class SumTree:                                                                                 # PER санах ойны SumTree бүтэц 
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

class PERMemory:                                                                               # PER санах ой 
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


class CommonNetwork(flax.nn.Module):                                                           # Common Network 
    def apply(self, x, n_outputs):
        dense_layer_1      = flax.nn.Dense(x, 256) 
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 256) 
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_outputs)
        return output_layer


class TwinQNetwork(flax.nn.Module): # Twin Q Network 
    def apply(self, x):
        q1 = CommonNetwork(x, 1)
        q2 = CommonNetwork(x, 1)
        return q1, q2


# https://github.com/google/jax/issues/2173 Gaussian Normal distribution 
# https://github.com/pytorch/pytorch/blob/master/torch/distributions/normal.py#L65
@jax.jit
def gaussian_normal(key, mu, sigma):
    normals = mu + jax.random.normal(key, mu.shape)*sigma
    return normals

#https://github.com/pytorch/pytorch/blob/master/torch/distributions/normal.py#L70 Log probability calculation 
@jax.jit
def log_prob(mean, scale, value):
    var       = (jnp.square(scale))
    log_scale = jnp.log(scale)
    return -((value - mean) ** 2) / (2 * var) - log_scale - jnp.log(jnp.sqrt(2 * jnp.pi))

class GaussianPolicy(flax.nn.Module):                                                          # Gaussian Policy Network 
    def apply(self, x, n_actions, key=None, sample=False, clip_min=-2.0, clip_max=2.0):        # clip range-ийг тохируулах
        policy_layer  = CommonNetwork(x, n_actions*2)
        mean, log_std = jnp.split(policy_layer, 2, axis=-1)
        log_std       = jnp.clip(log_std, clip_min, clip_max)                                  # clip log_std range
        std           = jnp.exp(log_std)                                                       # std dev-ийг log_std-ээс гаргаж авах
        if sample:
            xs        = gaussian_normal(key, mean, std)
            actions   = flax.nn.tanh(xs)                                                       # tanh ашиглан action range-ийг [-1, 1] болгох
            log_probs = log_prob(mean, std, xs) - jnp.log(1-jnp.square(actions)+1e-6)          # Change 1e-5 to 1e-6 for numerical stability
            entropies = -jnp.sum(log_probs, axis=-1, keepdims=True)                            # sum over action dimensions
            return actions, entropies, flax.nn.tanh(mean)                                      # tanh(mean)-ийг буцаах
        else:
            return mean, std # std-ийг буцаах


# https://github.com/henry-prior/jax-rl/blob/436b009cd97475b75be3b192a0ba761152950f41/models.py#L98 Constant for alpha 
class Constant(flax.nn.Module):
    def apply(self, start_value, dtype=jnp.float32):
        value = self.param('value', (1,), flax.nn.initializers.ones)
        return start_value * jnp.asarray(value, dtype)


environment_name = 'MountainCarContinuous-v0'                                                  # Environment name 
env              = gym.make(environment_name)

if debug_render:                                                                               # Render if debug_render is True 
    env.render(mode="human")
state = env.reset()


clip_min=-2.0                                                                                  # clip_min and clip_max for log_std 
clip_max= 2.0                                                                                  # clip_min and clip_max for log_std 


critic_module = TwinQNetwork.partial()                                                         # Critic module 
_, params     = critic_module.init_by_shape(                                                   # Critic params 
    jax.random.PRNGKey(0),
    [(env.observation_space.shape[0]+env.action_space.shape[0],)])                             # state + action input shape

critic        = flax.nn.Model(critic_module, params)                                           # Critic model 
target_critic = flax.nn.Model(critic_module, params)                                           # Target critic model 

actor_module    = GaussianPolicy.partial(                                                      # Actor module ( clip range-ийг тохируулах)
    n_actions   = env.action_space.shape[0],
    key         = jax.random.PRNGKey(0),
    clip_min    = clip_min,
    clip_max    = clip_max
    )
_, actor_params = actor_module.init_by_shape(                                                  # Actor params 
    jax.random.PRNGKey(0),
    [env.observation_space.shape])                                                             # state input shape
actor           = flax.nn.Model(actor_module, actor_params)                                    # Actor model 

constant_module    = Constant.partial(start_value=alpha)                                       # Alpha constant module 
_, constant_params = constant_module.init(jax.random.PRNGKey(0))                               # Alpha constant params 
constant_model     = flax.nn.Model(constant_module, constant_params)                           # Alpha constant model 
target_entropy     = -env.action_space.shape[0]                                                # Target entropy 


critic_optimizer = flax.optim.Adam(learning_rate).create(critic)                               # Critic optimizer 
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor)                                # Actor optimizer 
alpha_optimizer  = flax.optim.Adam(learning_rate).create(constant_model)                       # Alpha optimizer 



@jax.jit                                                                                       # Actor inference 
def actor_inference(model, state, key):
    actions, entropies, tanh_means = model(state, key=key, sample=True)                        # sample=True ашиглах
    return actions, entropies, tanh_means

@jax.jit                                                                                       # Critic inference 
def critic_inference(model, state, action):
    state_action = jnp.concatenate((state, action), axis=-1)                                   # state and action concatenation
    q1, q2       = model(state_action)                                                         # twin q values
    return q1, q2

@jax.jit                                                                                       # Target Critic inference 
def target_critic_inference(actor_model, target_critic_model, next_state, reward, done, alpha, key):
    next_actions, next_entropies, _ = actor_model(next_state, key=key, sample=True)            # sample next action from actor
    next_state_action               = jnp.concatenate((next_state, next_actions), axis=-1)     # concatenate next state and next action
    next_q1, next_q2                = target_critic_model(next_state_action)                   # get twin q values for next state-action
    min_q                           = jnp.minimum(next_q1, next_q2)                            # minimum of twin q values
    next_q                          = min_q-alpha*next_entropies                               # subtract entropy * alpha
    rewards                         = jnp.reshape(reward, (reward.shape[0], 1))                # reshape reward
    dones                           = jnp.subtract(1.0, done)                                  # 1.0 - done
    dones                           = jnp.reshape(dones, (dones.shape[0], 1))                  # reshape dones
    target_q                        = rewards+gamma*next_q*dones                               # TD target calculation
    return target_q


# https://github.com/henry-prior/jax-rl/blob/436b009cd97475b75be3b192a0ba761152950f41/utils.py#L52 Double MSE loss 
@jax.vmap
def double_mse(q1, q2, qt): # vmap for batch mse loss
    return jnp.square(qt-q1).mean() + jnp.square(qt-q2).mean()

@jax.jit # Critic backpropagation 
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

    target_q = jax.lax.stop_gradient(                                                          # stop gradient for target q value
        target_critic_inference(                                                               # target critic inference
            actor_optimizer.target,
            target_critic,
            batch[3],
            batch[2],
            batch[4],
            alpha,
            key
        )
    )

    def loss_fn(critic_model):                                                                 # critic loss function
        state_action = jnp.concatenate((batch[0], batch[1]), axis=-1)                          # state action concatenation
        q1, q2       = critic_model(state_action)                                              # twin q values
        critic_loss  = double_mse(q1, q2, target_q)                                            # double mse loss
        return jnp.mean(critic_loss)                                                           # mean over batch
    loss, gradients  = jax.value_and_grad(loss_fn)(critic_optimizer.target)                    # value and grad
    critic_optimizer = critic_optimizer.apply_gradient(gradients)                              # apply gradients

    state_action_noclip = jnp.concatenate((batch[0], batch[1]), axis=-1)                       # state action concatenation for td_error
    q1_noclip, _ = jax.lax.stop_gradient(critic_inference(critic_optimizer.target, state_action_noclip)) # q value without clip for td_error
    td_errors = jnp.abs(q1_noclip-target_q)                                                    # td error calculation

    return critic_optimizer, td_errors                                                         # return optimizer and td_errors


@jax.jit # Actor backpropagation 
def backpropagate_actor(optimizer, critic, batch, alpha, key):
    # batch[0] - states
    # batch[5] - weights
    def loss_fn(actor_model):                                                                  # actor loss function
        sampled_actions, entropies, _ = actor_inference(                                       # actor inference for sampled actions
            optimizer.target,
            batch[0],
            key
            )
        q1, q2 = critic_inference(                                                             # critic inference for sampled actions
            critic,
            batch[0],
            sampled_actions
            )
        q_values   = jnp.minimum(q1, q2)                                                       # minimum of twin q values
        actor_loss = jnp.mean(                                                                 # actor loss calculation
            (-q_values + entropies*alpha)                                                      # subtract q values and add entropy * alpha
        )
        return actor_loss, -entropies                                                          # return actor loss and entropies

    gradients, entropies = jax.grad(loss_fn, has_aux=True)(optimizer.target)                   # value and grad, get entropies
    actor_optimizer      = optimizer.apply_gradient(gradients)                                 # apply gradients

    return actor_optimizer, entropies                                                          # return optimizer and entropies

# https://github.com/henry-prior/jax-rl/blob/436b009cd97475b75be3b192a0ba761152950f41/utils.py#L43 Soft update function 
@jax.jit
def soft_update(model, target_model, tau):
    update_params = jax.tree_multimap(
        lambda m1, mt: tau*m1 + (1-tau)*mt,
        model.params, target_model.params
        )
    return target_model.replace(params=update_params)


@jax.jit # Alpha backpropagation 
def backpropagate_alpha(optimizer, entropies, weights, target_entropy):
    entropies = jax.lax.stop_gradient(entropies)                                               # stop gradient for entropies
    weights   = jnp.reshape(weights, (weights.shape[0], 1))                                    # reshape weights
    # entopy нь target entropy-ээс бага байвал alpha-г нэмнэ
    # эсрэгээрээ их байвал alpha-г бууруулна
    def loss_fn(constant_model):                                                               # alpha loss function
        log_alpha   = constant_model()                                                         # get log alpha
        alpha       = jnp.exp(log_alpha)                                                       # alpha from log alpha
        alpha_loss  = jnp.mean(                                                                # alpha loss calculation
            -log_alpha * jax.lax.stop_gradient(entropies + target_entropy)                     # -log_alpha * (entropy + target_entropy)
        )
        return alpha_loss, alpha                                                               # return alpha loss and alpha

    gradients, alpha = jax.grad(loss_fn, has_aux=True)(optimizer.target)                       # value and grad, get alpha
    alpha_optimizer  = optimizer.apply_gradient(gradients)                                     # apply gradients
    return alpha_optimizer, alpha                                                              # return optimizer and alpha


per_memory   = PERMemory(memory_length)                                                        # PER memory 
rng          = jax.random.PRNGKey(0)                                                           # random key 

global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []                                                                   # episode rewards list
        state = env.reset()                                                                    # reset environment
        epsilon = epsilon_max                                                                  # reset epsilon at the start of episode
        for episode_step in range(500):                                                        # limit episode steps for MountainCarContinuous-v0
            global_steps = global_steps+1                                                      # increment global steps

            alpha_value         = jnp.exp(alpha_optimizer.target)[0]                           # alpha value from optimizer

            if np.random.rand() <= epsilon:                                                    # exploration with epsilon-greedy 
                action = env.action_space.sample()                                             # random action
            else:
                rng, new_key  = jax.random.split(rng)                                          # split random key
                actions, _, _ = actor_inference(                                               # actor inference for action
                    actor,
                    jnp.asarray([state]),
                    new_key)
                action = actions[0]                                                            # get action from actor output

            if epsilon>epsilon_min:                                                            # epsilon decay ( decay rate-ийг багасгах)
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps) # decay epsilon

            next_state, reward, done, _ = env.step(action)                                     # environment step

            # sample цуглуулж жинг тооцон PER санах ойруу хийх 
            q1, q2 = critic_inference(                                                         # critic inference for current state-action
                critic,
                jnp.asarray([state]),
                jnp.asarray([action])
                )
            rng, new_key = jax.random.split(rng)                                               # split random key
            target_q     = jax.lax.stop_gradient(                                              # stop gradient for target q
                target_critic_inference(                                                       # target critic inference for next state
                    actor,
                    target_critic,
                    jnp.asarray([next_state]),
                    jnp.asarray([reward]),
                    jnp.asarray([int(done)]),
                    alpha_value,
                    new_key
                )
            )
            td_error = jnp.abs(q1-target_q)                                                    # td error calculation
            td_error = td_error[0][0]                                                          # get td error value
            per_memory.add(td_error, (state, action, reward, next_state, int(done)))           # add sample to PER memory

            episode_rewards.append(reward)                                                     # append reward to episode rewards

            if global_steps > batch_size*10: # batch size-ээс дээш алхам хийсний дараа сургалт хийнэ, start training after buffer is filled a bit
                # сургах batch бэлтгэх 
                batch = per_memory.sample(batch_size)                                          # sample batch from PER memory
                states, actions, rewards, next_states, dones, weights = [], [], [], [], [], [] # lists for batch data
                for i in range(batch_size):                                                    # loop over batch size
                    weights.append    (batch[i][1])                                            # weights from PER sample
                    states.append     (batch[i][1][2][0])                                      # state from PER sample
                    actions.append    (batch[i][1][2][1])                                      # action from PER sample
                    rewards.append    (batch[i][1][2][2])                                      # reward from PER sample
                    next_states.append(batch[i][1][2][3])                                      # next state from PER sample
                    dones.append      (batch[i][1][2][4])                                      # done from PER sample


                # critic неорон сүлжээг сургах 
                rng, new_key = jax.random.split(rng)                                           # split random key
                critic_optimizer, td_errors = backpropagate_critic(                            # critic backpropagation
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
                    alpha_value,
                    new_key
                    )

                # PER санах ойны жинг шинэчлэх 
                new_td_errors = np.array(td_errors).flatten()                                  # flatten td errors
                for i in range(batch_size):                                                    # loop over batch size
                    idx = batch[i][0]                                                          # get index from PER sample
                    per_memory.update(idx, new_td_errors[i])                                   # update PER memory with new td errors

                # actor неорон сүлжээг сургах 
                rng, new_key = jax.random.split(rng)                                           # split random key
                actor_optimizer, entropies = backpropagate_actor(                              # actor backpropagation
                    actor_optimizer,
                    critic_optimizer.target,
                    (
                        jnp.asarray(states),
                        jnp.asarray(weights)
                    ),
                    alpha_value,
                    new_key
                    )

                # alpha утгыг сургах 
                rng, new_key = jax.random.split(rng)                                           # split random key
                alpha_optimizer, alpha_value = backpropagate_alpha(                            # alpha backpropagation
                    alpha_optimizer,
                    entropies,
                    jnp.asarray(weights),
                    target_entropy
                    )


                # сайжирсэн жингүүдээр target_critic неорон сүлжээг шинэчлэх 
                if global_steps%sync_steps==0:                                                 # soft update target critic every sync_steps
                    target_critic = soft_update(critic_optimizer.target, target_critic, tau)   # soft update target critic


            state = next_state # next state for next step

            if debug_render:                                                                   # render if debug_render is True
                #time.sleep(1. / 60)
                env.render(mode="human")

            if done or episode_step == 499:                                                    # episode done or max steps reached
                print("{} - нийт reward : {}, alpha: {:.2f}, epsilon: {:.2f}".format(          # print episode info
                    episode,
                    sum(episode_rewards),
                    alpha_value,
                    epsilon
                    ))
                break
finally:
    env.close() 