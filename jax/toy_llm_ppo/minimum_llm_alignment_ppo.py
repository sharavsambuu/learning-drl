#
#
#  The goal is to teach a little LLM to write happy sentences.
#
#
# Our Reward Model is like a "positivity meter" for sentences
# It doesn't compare our robot's sentence to some perfect "positive sentence." Instead, 
# it just reads the sentence and gives it a score based on how "positive" it sounds, 
# according to its own simple rules.
#
# Let's say our "positive keywords" are: "good", "happy".
#
#   Generated Sentence 1: "This is a good day."
#   Reward Model: Counts "good" (1 keyword).
#   Reward Score: 1
#
#   Generated Sentence 2: "Today is happy and sunny."
#   Reward Model: Counts "happy" (1 keyword).
#   Reward Score: 1
#
#   Generated Sentence 3: "This is a great and wonderful day!"
#   Reward Model: Counts "great", "wonderful" (2 keywords).
#   Reward Score: 2
#
#   Generated Sentence 4: "The weather is bad."
#   Reward Model: Counts no positive keywords.
#   Reward Score: 0
#
#
# In our character-level Toy LLM:
#
#   The "action" is the choice of a character to generate next.
#
#
#z
#


import os
import random
import math
import gym
from collections import deque

import flax
import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time


debug_render    = False
num_episodes    = 1000
learning_rate   = 0.001
gamma           = 0.99
gae_lambda      = 0.95
clip_ratio      = 0.2
policy_epochs   = 10
batch_size      = 32
mini_batch_size = 16
sync_steps      = 100
sentence_length = 20
vocab_size      = 32 # Defined below

vocabulary_characters = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ' ', '.', ',', '!', '?', '<EOS>'
]
char_to_index = {char: index for index, char in enumerate(vocabulary_characters)}
vocab_size = len(vocabulary_characters)
eos_token_index = char_to_index['<EOS>']


class ToyLLM(flax.nn.Module): # Toy LLM - Refined apply method to use carry for output
    n_vocab: int
    embedding_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, carry):
        embed = nn.Embed(num_embeddings=self.n_vocab, features=self.embedding_dim, embedding_init=jax.nn.initializers.uniform())(x)
        gru_cell = nn.GRUCell(features=self.hidden_size)
        _, carry = gru_cell(carry, embed) # Update carry state
        output_logits = nn.Dense(features=self.n_vocab)(carry) # Output logits from carry state
        return output_logits, carry

    @staticmethod
    def initial_state(batch_size: int, hidden_size: int): # Initial state helper 
        return nn.GRUCell(features=hidden_size).initialize_carry(jax.random.PRNGKey(0), (batch_size,), hidden_size)


class CriticNetwork(flax.nn.Module): # Critic Network 
    def apply(self, x):
        dense_layer_1      = flax.nn.Dense(x, 32)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        output_dense_layer = flax.nn.Dense(activation_layer_1, 1)
        return output_dense_layer


# Initialize global networks and optimizers 
toy_llm_module   = ToyLLM.partial(n_vocab=vocab_size, embedding_dim=32, hidden_size=64)
initial_carry    = toy_llm_module.initial_state(batch_size=1, hidden_size=64)
_, actor_params  = toy_llm_module.init(jax.random.PRNGKey(0), jnp.zeros((1,), dtype=jnp.int32), initial_carry)
actor_model      = flax.nn.Model(toy_llm_module, actor_params)
actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)

critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [(64,)])
critic_model     = flax.nn.Model(critic_module, critic_params)
critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model)


@jax.jit
def actor_inference(model, x, carry): # Inference function for Toy LLM 
    logits, carry = model(x, carry, method='apply')
    action_probabilities = nn.softmax(logits)
    return action_probabilities, carry

@jax.jit
def critic_inference(model, x): # Critic inference 
    return model(x)


def reward_model(sentence): # Synthetic Reward Model 
    positive_keywords = ["good", "great", "wonderful", "amazing", "happy", "joyful", "positive"]
    score           = 0
    for keyword in positive_keywords:
        if keyword in sentence.lower():
            score += 1
    return score


# Generalized Advantage Estimation (GAE) function
def gae_advantage(rewards, values, last_value, gamma, gae_lambda): 
    values_np     = np.array(values + [last_value])
    advantages    = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        delta         = rewards[t] + gamma * values_np[t + 1] - values_np[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
    return advantages

@jax.jit
def train_step(actor_optimizer, critic_optimizer, actor_model, critic_model, batch, initial_carry): # Train step function 
    states, actions, old_log_probs, advantages, returns, carries = batch

    def actor_loss_fn(actor_model):
        action_probabilities, _ = actor_model(states, carries, method='apply')
        log_probs = jnp.log(action_probabilities[jnp.arange(len(actions)), actions])
        ratio = jnp.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        return actor_loss

    def critic_loss_fn(critic_model):
        values = critic_model(states).reshape(-1)
        critic_loss = jnp.mean((values - returns)**2)
        return critic_loss

    actor_grads, actor_loss   = jax.value_and_grad(actor_loss_fn)(actor_optimizer.target)
    critic_grads, critic_loss = jax.value_and_grad(critic_loss_fn)(critic_optimizer.target)

    actor_optimizer  = actor_optimizer.apply_gradient(actor_grads)
    critic_optimizer = critic_optimizer.apply_gradient(critic_grads)

    return actor_optimizer, critic_optimizer, actor_loss, critic_loss


def decode_text(token_indices): # Decoder function 
    decoded_text = ""
    for index in token_indices:
        if index == eos_token_index:
            break
        decoded_text += vocabulary_characters[index]
    return decoded_text


rng = jax.random.PRNGKey(0)
try:
    for episode in range(num_episodes):
        state_carry        = initial_carry # Reset RNN carry state at start of each episode 
        episode_rewards    = 0
        episode_sentences  = []
        episode_states, episode_actions, episode_rewards_list, episode_log_probs, episode_values, episode_carries = [], [], [], [], [], []

        sentence_tokens    = []

        for step in range(sentence_length):
            rng, action_key, value_key = jax.random.split(rng, 3)

            # Inference - Pass carry state sequentially
            action_probabilities, next_carry = actor_inference(actor_model, jnp.asarray([sentence_tokens[-1:] if sentence_tokens else [0]]), state_carry)

            action_probabilities = np.array(action_probabilities[0])
            action_index         = np.random.choice(vocab_size, p=action_probabilities)
            predicted_value      = critic_inference(critic_optimizer.target, jnp.asarray([state_carry[0]]))
            log_prob             = jnp.log(action_probabilities[action_index])

            action = int(action_index)

            sentence_tokens.append(action)
            text_state = tuple(sentence_tokens) # State is still token sequence

            episode_states.append(text_state)
            episode_actions.append(action)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_carries.append(state_carry) # Store carry state for *this step* 

            state_carry = next_carry # Update carry state for *next* step 


            if action_index == eos_token_index:
                done = True
                break

            episode_rewards_list.append(0)
            state = text_state


        sentence_text   = decode_text(sentence_tokens)
        reward          = reward_model(sentence_text)
        episode_rewards = reward

        advantages = gae_advantage(episode_rewards_list, episode_values, 0, gamma, gae_lambda)
        returns    = advantages + np.array(episode_values)


        batch_data = (
            np.array(episode_states),
            np.array(episode_actions),
            np.array(episode_log_probs),
            advantages,
            returns,
            np.array([carry[0] for carry in episode_carries]) # Extract carry arrays
        )


        for _ in range(policy_epochs):
            perm = np.random.permutation(len(episode_states))
            for start_idx in range(0, len(episode_states), mini_batch_size):
                mini_batch_idx = perm[start_idx:start_idx + mini_batch_size]
                mini_batch = tuple(arr[mini_batch_idx] for arr in batch_data)
                actor_optimizer, critic_optimizer, actor_loss, critic_loss = train_step(
                    actor_optimizer, critic_optimizer, actor_model, critic_model, mini_batch, initial_carry
                )


        print(f"Episode {episode}, Reward: {episode_rewards:.2f}, Sentence: '{sentence_text}'")

finally:
    pass