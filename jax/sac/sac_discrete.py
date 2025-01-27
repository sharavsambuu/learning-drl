import os
import random
import math
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np
import numpy

debug_render               = False  # render-ийг debug хийх үед True болгоно уу
debug                      = False
num_episodes               = 1500
learning_rate              = 0.0003 # learning rate-ийг багасгах нь сургалтыг тогтворжуулна
gamma                      = 0.99
tau                        = 0.005  # target network-ийг зөөлөн шинэчлэх параметр
target_entropy_coefficient = 0.98   # зорилтот энтропийн коэффициент, action space-ийн хэмжээтэй ойролцоо байх нь зүйтэй

class ActorNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 256) # давхаргын хэмжээг нэмэгдүүлэх
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 256) # давхаргын хэмжээг нэмэгдүүлэх
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer) # discrete action space-д softmax ашиглана
        return output_layer

class CriticNetwork(flax.nn.Module):
    def apply(self, x):
        dense_layer_1      = flax.nn.Dense(x, 256) # давхаргын хэмжээг нэмэгдүүлэх
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 256) # давхаргын хэмжээг нэмэгдүүлэх
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
        return output_dense_layer


env   = gym.make('CartPole-v1')
state = env.reset()
n_actions        = env.action_space.n

# Actor сүлжээ
actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model      = flax.nn.Model(actor_module, actor_params)
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)

# Critic сүлжээ 1
critic_module_1       = CriticNetwork.partial()
_, critic_params_1    = critic_module_1.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model_1        = flax.nn.Model(critic_module_1, critic_params_1)
critic_optimizer_1    = flax.optim.Adam(learning_rate).create(critic_model_1)
# Critic сүлжээ 2
critic_module_2       = CriticNetwork.partial()
_, critic_params_2    = critic_module_2.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model_2        = flax.nn.Model(critic_module_2, critic_params_2)
critic_optimizer_2    = flax.optim.Adam(learning_rate).create(critic_model_2)

# Зорилтот Critic сүлжээ 1, 2 (эхэндээ онлайн Critic сүлжээтэй ижил жинтэй)
target_critic_model_1 = flax.nn.Model(critic_module_1, critic_params_1)
target_critic_model_2 = flax.nn.Model(critic_module_2, critic_params_2)

# Энтропийн температурыг сургах параметр (log scale дээр сургана, учир нь alpha>0 байх ёстой)
log_alpha           = jnp.zeros(())
alpha_optimizer     = flax.optim.Adam(learning_rate).create(log_alpha)
target_entropy      = -target_entropy_coefficient * n_actions # зорилтот энтропи, ойролцоогоор -log(1/n_actions)


@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def critic_inference(model, x):
    return model(x)

@jax.jit
def backpropagate_critic(
        critic_optimizer,
        critic_model,
        target_critic_model_1,
        target_critic_model_2,
        actor_model,
        alpha,
        batch_props
    ):
    # batch_props[0] - states
    # batch_props[1] - next_states
    # batch_props[2] - rewards
    # batch_props[3] - dones
    # batch_props[4] - actions
    next_action_probabilities = actor_model(batch_props[1])
    next_log_action_probabilities = jnp.log(next_action_probabilities)
    next_q_values_1           = target_critic_model_1(batch_props[1])
    next_q_values_2           = target_critic_model_2(batch_props[1])
    next_min_q_values         = jnp.minimum(next_q_values_1, next_q_values_2)
    next_values               = next_action_probabilities * (next_min_q_values - alpha * next_log_action_probabilities)
    next_values               = jnp.sum(next_values, axis=1, keepdims=True) # action-уудын хэмжээгээр sum хийх
    target_q_values           = batch_props[2] + gamma*(1-batch_props[3])*next_values[:, 0] # batch dimension-ийг хасах

    def loss_fn(critic_model):
        current_q_values = critic_model(batch_props[0])
        critic_loss      = jnp.mean(jnp.square(target_q_values - current_q_values[:, 0])) # MSE loss
        return critic_loss, { 'critic_loss' : critic_loss } # loss-ыг metrics-д хадгалах
    (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(critic_model) # metrics-ийг авахын тулд has_aux=True болгох
    critic_optimizer            = critic_optimizer.apply_gradient(gradients)
    return critic_optimizer, metrics

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(actor_optimizer, critic_model_1, critic_model_2, alpha, batch_props):
    # batch_props[0] - states
    # batch_props[4] - actions
    def loss_fn(actor_model):
        action_probabilities      = actor_model(batch_props[0])
        log_action_probabilities  = jnp.log(action_probabilities)
        q_values_1                = critic_model_1(batch_props[0])
        q_values_2                = critic_model_2(batch_props[0])
        min_q_values              = jnp.minimum(q_values_1, q_values_2)
        actor_loss                = jnp.mean(
            (alpha * log_action_probabilities - min_q_values) * action_probabilities
        ) # Policy Gradient loss-г entropy-той хослуулсан
        return actor_loss, { 'actor_loss' : actor_loss, 'entropy': -jnp.mean(jnp.sum(action_probabilities * log_action_probabilities, axis=1))} # metrics-д entropy-г хадгалах
    (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(actor_model) # metrics-ийг авахын тулд has_aux=True болгох
    actor_optimizer           = actor_optimizer.apply_gradient(gradients)
    return actor_optimizer, metrics

@jax.jit
def backpropagate_alpha(alpha_optimizer, actor_model, batch_props, target_entropy):
    # batch_props[0] - states
    def loss_fn(log_alpha):
        alpha                 = jnp.exp(log_alpha)
        action_probabilities  = actor_model(batch_props[0])
        log_action_probabilities = jnp.log(action_probabilities)
        entropy               = -jnp.sum(action_probabilities * log_action_probabilities, axis=1)
        alpha_loss            = jnp.mean(-log_alpha * jax.lax.stop_gradient(entropy + target_entropy))
        return alpha_loss, {'alpha_loss': alpha_loss, 'alpha': alpha} # metrics-д alpha-г хадгалах
    (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(alpha_optimizer.target) # metrics-ийг авахын тулд has_aux=True болгох
    alpha_optimizer           = alpha_optimizer.apply_gradient(gradients)
    return alpha_optimizer, metrics

@jax.jit
def soft_update(target_model, online_model, tau):
    new_params = jax.tree_util.tree_map(
        lambda target_params, online_params: (1 - tau) * target_params + tau * online_params,
        target_model.params, online_model.params
    )
    return target_model.replace(params=new_params)


replay_buffer_states      = np.empty((memory_size, env.observation_space.shape[0]), dtype=state.dtype) # санах ойн хэмжээ, state-ийн dtype-тэй ижил
replay_buffer_actions     = np.empty((memory_size, 1          ), dtype=np.int32    ) # action нь integer утгатай
replay_buffer_rewards     = np.empty((memory_size, 1          ), dtype=np.float32  )
replay_buffer_next_states = np.empty((memory_size, env.observation_space.shape[0]), dtype=state.dtype) # санах ойн хэмжээ, state-ийн dtype-тэй ижил
replay_buffer_dones       = np.empty((memory_size, 1          ), dtype=np.float32  )
memory_counter            = 0
memory_size               = 10000 # replay buffer-ийн хэмжээ
batch_size                = 256   # batch size-ийг нэмэгдүүлэх

global_step = 0
try:
    for episode in range(num_episodes):
        state           = env.reset()
        episode_rewards = []
        while True:
            global_step = global_step+1

            alpha_value         = jnp.exp(alpha_optimizer.target) # alpha утгыг авах
            action_probabilities  = actor_inference(actor_optimizer.target, jnp.asarray([state]))
            action_probabilities  = np.array(action_probabilities[0])
            action                = np.random.choice(n_actions, p=action_probabilities) # магадлалаар sample-дах

            next_state, reward, done, _ = env.step(int(action))

            # Replay Buffer-т хадгалах
            idx                           = memory_counter % memory_size
            replay_buffer_states     [idx] = state
            replay_buffer_actions    [idx] = action
            replay_buffer_rewards    [idx] = reward
            replay_buffer_next_states[idx] = next_state
            replay_buffer_dones      [idx] = done
            memory_counter                += 1

            episode_rewards.append(reward)

            if global_step > batch_size: # batch size-ээс дээш алхам хийсний дараа сургалт хийнэ
                # Replay Buffer-оос batch sample-дах
                indices = np.random.choice(memory_size, size=batch_size)
                batch_states      = replay_buffer_states     [indices]
                batch_actions     = replay_buffer_actions    [indices]
                batch_rewards     = replay_buffer_rewards    [indices]
                batch_next_states = replay_buffer_next_states[indices]
                batch_dones       = replay_buffer_dones      [indices]

                # Critic сүлжээ 1-ийг сургах
                critic_optimizer_1, critic_metrics_1 = backpropagate_critic(
                    critic_optimizer_1,
                    critic_model_1,
                    target_critic_model_1,
                    target_critic_model_2,
                    actor_optimizer.target,
                    alpha_value,
                    (
                        jnp.asarray(batch_states),
                        jnp.asarray(batch_next_states),
                        jnp.asarray(batch_rewards),
                        jnp.asarray(batch_dones),
                        jnp.asarray(batch_actions).reshape(batch_size,) # actions-ийг reshape хийх
                    )
                )
                # Critic сүлжээ 2-ыг сургах
                critic_optimizer_2, critic_metrics_2 = backpropagate_critic(
                    critic_optimizer_2,
                    critic_model_2,
                    target_critic_model_1,
                    target_critic_model_2,
                    actor_optimizer.target,
                    alpha_value,
                    (
                        jnp.asarray(batch_states),
                        jnp.asarray(batch_next_states),
                        jnp.asarray(batch_rewards),
                        jnp.asarray(batch_dones),
                        jnp.asarray(batch_actions).reshape(batch_size,) # actions-ийг reshape хийх
                    )
                )
                # Actor сүлжээг сургах
                actor_optimizer, actor_metrics = backpropagate_actor(
                    actor_optimizer,
                    critic_model_1,
                    critic_model_2,
                    alpha_value,
                    (
                        jnp.asarray(batch_states),
                        jnp.asarray(batch_actions).reshape(batch_size,) # actions-ийг reshape хийх
                    )
                )
                # Alpha-г сургах
                alpha_optimizer, alpha_metrics = backpropagate_alpha(
                    alpha_optimizer,
                    actor_optimizer.target,
                    (jnp.asarray(batch_states)),
                    target_entropy
                )

                # Target Critic сүлжээг зөөлөн шинэчлэх
                target_critic_model_1 = soft_update(target_critic_model_1, critic_model_1, tau)
                target_critic_model_2 = soft_update(target_critic_model_2, critic_model_2, tau)


            state = next_state

            if debug_render:
                env.render()

            if done:
                print(episode, " - reward :", sum(episode_rewards),
                      "- critic_loss_1: ", critic_metrics_1['critic_loss'],
                      "- critic_loss_2: ", critic_metrics_2['critic_loss'],
                      "- actor_loss: ", actor_metrics['actor_loss'],
                      "- alpha_loss: ", alpha_metrics['alpha_loss'],
                      "- entropy: ", actor_metrics['entropy'],
                      "- alpha: ", alpha_metrics['alpha'])
                break
finally:
    env.close()