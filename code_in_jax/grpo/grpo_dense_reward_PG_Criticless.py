#
# GRPO (Critic-less) + Policy Gradient (REINFORCE) + Outcome-Based Group Advantage + Entropy Bonus
#
#    - Group rollouts (K trajectories) per update
#    - Critic-less:                                    no V(s), no bootstrapping, no value loss
#    - Outcome-based group advantage (DeepSeek-style): per-trajectory scalar A_i = R_i - mean(R)   (NO std division)
#                                                      expanded to every step in that trajectory
#    - Actor update (policy gradient):                 -E[ log_pi(a|s) * stopgrad(A_step) + ent_coef * H(pi) ]
#
#

import os
import random
import math
import jax
import optax
import numpy      as np
import gymnasium  as gym
import flax.linen as nn
from   jax        import numpy as jnp


debug_render   = True
debug          = True
play_frequency = 30
num_episodes   = 10000
learning_rate  = 0.001
gamma          = 0.99
env_name       = "CartPole-v1"
group_size     = 6
max_steps      = 500

entropy_coefficient = 0.01


class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=self.n_actions)(activation_layer_2)
        output_layer       = nn.softmax(output_dense_layer)
        return output_layer


env_array   = [gym.make(env_name, render_mode=None) for _ in range(group_size)]
env         = gym.make(env_name, render_mode='human')

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n


actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

actor_optimizer_def    = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params)


@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, props):
    # props[0] - states     (B, S)
    # props[1] - actions    (B,)
    # props[2] - advantages (B,)  
    def loss_fn(params):
        action_probas = actor_module.apply({'params': params}, props[0])

        probs         = jnp.take_along_axis(action_probas, props[1][:, None], axis=1).squeeze(1)
        probs         = jnp.clip(probs, 1e-8, 1.0)
        logp          = jnp.log(probs)

        # Critic-less GRPO: advantage is provided externally (group outcome scalar expanded to steps)
        advantages_sg = jax.lax.stop_gradient(props[2])

        entropies     = -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1)

        #   - maximize logp * advantage  (=> minimize -logp*adv)
        #   - add entropy bonus          (=> subtract ent_coef * entropy from loss)
        pg_loss       = -jnp.mean(logp * advantages_sg)
        ent_bonus     =  jnp.mean(entropies)

        loss          = pg_loss - entropy_coefficient * ent_bonus
        return loss, (pg_loss, ent_bonus)

    (loss, (pg_loss, ent_bonus)), gradients = jax.value_and_grad(loss_fn, has_aux=True)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_params, loss, pg_loss, ent_bonus


def rollout_trajectory(group_member_id, actor_model_params, seed):
    env_item       = env_array[group_member_id]

    # each group member starts from the same seeded reset each episode
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states         = np.zeros(shape=(max_steps,) + state_shape, dtype=np.float32)
    actions        = np.zeros(shape=(max_steps,), dtype=np.int32)
    rewards        = np.zeros(shape=(max_steps,), dtype=np.float32)

    step           = 0

    for _ in range(max_steps):
        action_probabilities = actor_inference(actor_model_params, jnp.asarray([state]))
        action_probabilities = np.array(action_probabilities[0])
        action               = np.random.choice(n_actions, p=action_probabilities)

        next_state, reward, terminated, truncated, info = env_item.step(int(action))

        done_boundary = bool(terminated or truncated)

        next_state = np.array(next_state, dtype=np.float32)

        states [step, :] = state
        actions[step   ] = int  (action   )
        rewards[step   ] = float(reward   )

        step += 1

        state = next_state

        if done_boundary:
            break

    trajectory_length = step

    # Outcome / episodic return (CartPole: equals survival steps since reward=+1/step)
    total_reward = float(np.sum(rewards[:trajectory_length])) if trajectory_length > 0 else 0.0

    return group_member_id, total_reward, trajectory_length, states, actions


def rollout_group(actor_model_params, seed):
    group_total_rewards = np.zeros(shape=(group_size,), dtype=np.float32)
    group_lengths       = np.zeros(shape=(group_size,), dtype=np.int32  )

    group_states        = []
    group_actions       = []

    for group_member_id in range(group_size):
        member_id, total_reward, trajectory_length, states, actions = rollout_trajectory(
            group_member_id     = group_member_id,
            actor_model_params  = actor_model_params,
            seed                = seed
        )

        group_total_rewards[member_id] = float(total_reward)
        group_lengths      [member_id] = int  (trajectory_length)

        group_states .append(states [:trajectory_length])
        group_actions.append(actions[:trajectory_length])

    # Group baseline, mean episodic return across the K rollouts
    group_mean_reward = float(np.mean(group_total_rewards))

    # DeepSeek-style outcome advantage (no std division)
    group_adv_scalar  = (group_total_rewards - group_mean_reward)

    # Expand scalar per-trajectory advantage to every step in that trajectory (critic-less GRPO)
    group_advantages  = []
    for member_id in range(group_size):
        length  = int(group_lengths[member_id])
        adv_trj = np.full(shape=(length,), fill_value=float(group_adv_scalar[member_id]), dtype=np.float32)
        group_advantages.append(adv_trj)

    return group_mean_reward, group_lengths, group_states, group_actions, group_advantages


try:
    global_step = 0

    for episode in range(num_episodes):

        group_mean_reward, group_lengths, group_states, group_actions, group_advantages = rollout_group(
            actor_model_params  = actor_model_params,
            seed                = episode
        )

        # Flatten variable-length trajectories into one batch (no padding leakage)
        flat_states     = np.concatenate(group_states    , axis=0) if len(group_states    ) > 0 else np.zeros((0,) + state_shape, dtype=np.float32)
        flat_actions    = np.concatenate(group_actions   , axis=0) if len(group_actions   ) > 0 else np.zeros((0,), dtype=np.int32)
        flat_advantages = np.concatenate(group_advantages, axis=0) if len(group_advantages) > 0 else np.zeros((0,), dtype=np.float32)

        actor_loss = 0.0
        pg_loss    = 0.0
        ent_bonus  = 0.0

        if flat_states.shape[0] > 0:
            global_step += int(flat_states.shape[0])

            actor_optimizer_state, actor_model_params, actor_loss, pg_loss, ent_bonus = backpropagate_actor(
                actor_optimizer_state,
                actor_model_params,
                (
                    jnp.asarray(flat_states    , dtype=jnp.float32),
                    jnp.asarray(flat_actions   , dtype=jnp.int32  ),
                    jnp.asarray(flat_advantages, dtype=jnp.float32),
                )
            )

        if debug and (episode % 10 == 0):
            print(f"Ep {episode:6d} | Steps {global_step:9d} | "
                  f"GroupMeanR {group_mean_reward:9.3f} | "
                  f"ActLoss {float(actor_loss):9.4f} | PG {float(pg_loss):9.4f} | Ent {float(ent_bonus):9.4f}")

        if episode % play_frequency == 0 and debug_render == True:
            state, info = env.reset(seed=int(episode))
            state       = np.array(state, dtype=np.float32)

            rewards = []

            while True:
                action_probabilities  = actor_inference(actor_model_params, jnp.asarray([state]))
                action_probabilities  = np.array(action_probabilities[0])
                action                = np.random.choice(n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                next_state = np.array(next_state, dtype=np.float32)

                rewards.append(float(reward))
                state = next_state

                env.render()

                if done:
                    print(episode, " - reward :", sum(rewards))
                    break

finally:
    env.close()
    for env_item in env_array:
        env_item.close()
