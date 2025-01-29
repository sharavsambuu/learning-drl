#
#
# The goal is to teach a little LLM to write happy sentences.
#
#
# RL "State" => Text context: The "state" in RL terms represents the context of the sentence being generated. 
# It's how much of the sentence has been written so far.
#
# Critic's State => RNN's Hidden State: The Critic network uses the RNN's hidden state as 
# its "state" input to evaluate the sentence's progress.
#
# Actor's Action => Character Choice: The "action" the Toy LLM takes at each 
# step is choosing the next character to write from our vocabulary.
#
# PPO Teaches "Positive Writing": We use PPO to train the Toy LLM to generate 
# sentences that get high "positivity scores" from our simple Reward Model.
#
# Reward Model is like a "positivity meter" for sentences
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
#


import os
import random
import math
import gym
from collections import deque

import flax.linen as nn  # Use flax.linen for neural network definitions
import jax
from jax import numpy as jnp
import numpy as np
import optax
import time

# Hyperparameters
debug_render    = False
num_episodes    = 50000
learning_rate   = 0.001
gamma           = 0.99
gae_lambda      = 0.95  # GAE Lambda Hyperparameter
clip_ratio      = 0.2   # PPO Clip Ratio Hyperparameter
policy_epochs   = 10    # PPO Policy Epochs Hyperparameter
batch_size      = 32
mini_batch_size = 16
sentence_length = 20
hidden_size     = 128   # Hidden size for feedforward networks

# Vocabulary
vocabulary_characters = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ' ', '.', ',', '!', '?', '<EOS>'
]
char_to_index   = {char: index for index, char in enumerate(vocabulary_characters)}
vocab_size      = len(vocabulary_characters)
eos_token_index = char_to_index['<EOS>']


class ToyLLM(nn.Module):
    n_vocab: int

    @nn.compact
    def __call__(self, x): # Input is the state
        x = nn.Embed(num_embeddings=self.n_vocab, features=hidden_size, embedding_init=jax.nn.initializers.uniform())(x) # Embed input
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.n_vocab)(x)  # Output logits for vocabulary
        return logits

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x): # Input is the state
        x = nn.Embed(num_embeddings=vocab_size, features=hidden_size, embedding_init=jax.nn.initializers.uniform())(x) # Embed input
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        value = nn.Dense(features=1)(x) # Output a single value
        return value


# Initialize models and optimizers
rng = jax.random.PRNGKey(0)
toy_llm_module   = ToyLLM(n_vocab=vocab_size) # Pass vocab size to LLM
critic_module    = CriticNetwork()

dummy_input      = jnp.zeros((1,), dtype=jnp.int32) # Dummy input for init (single token index)
actor_params     = toy_llm_module.init(rng, dummy_input)['params'] # Access 'params' key
critic_params    = critic_module.init (rng, dummy_input)['params'] # Access 'params' key


actor_optimizer  = optax.adam(learning_rate)
critic_optimizer = optax.adam(learning_rate)

actor_opt_state  = actor_optimizer.init(actor_params)
critic_opt_state = critic_optimizer.init(critic_params)


@jax.jit
def actor_inference(actor_params, x):
    logits = toy_llm_module.apply({'params': actor_params}, x) # Pass params as dict
    action_probabilities = nn.softmax(logits)
    return action_probabilities

@jax.jit
def critic_inference(critic_params, x):
    value = critic_module.apply({'params': critic_params}, x) # Pass params as dict
    return value

def reward_model(sentence): # Synthetic Reward Model
    positive_keywords = ["good", "great", "wonderful", "amazing", "happy", "joyful", "positive"]
    score = 0
    for keyword in positive_keywords:
        if keyword in sentence.lower():
            score += 1
    return score

# Generalized Advantage Estimation (GAE) function
def gae_advantage(rewards, values, last_value, gamma, gae_lambda):
    values_np    = np.array(values + [last_value])
    advantages   = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        delta         = rewards[t] + gamma * values_np[t + 1] - values_np[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
    return advantages

@jax.jit
def train_step(actor_params, actor_opt_state, critic_params, critic_opt_state, batch):
    states, actions, old_log_probs, advantages, returns = batch

    def actor_loss_fn(actor_params):
        logits = toy_llm_module.apply({'params': actor_params}, states) # Pass params as dict
        action_probabilities = nn.softmax(logits)
        log_probs = jnp.log(action_probabilities[jnp.arange(len(actions)), actions])
        ratio = jnp.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        return actor_loss

    def critic_loss_fn(critic_params):
        values = critic_module.apply({'params': critic_params}, states).reshape(-1) # Pass params as dict
        critic_loss = jnp.mean((values - returns)**2)
        return critic_loss

    def combined_loss_fn(params): # Combined loss function
        actor_loss_val = actor_loss_fn(params[0]) # Calculate actor loss
        critic_loss_val = critic_loss_fn(params[1]) # Calculate critic loss
        combined_loss = actor_loss_val + critic_loss_val # Sum of actor and critic losses
        return combined_loss, (actor_loss_val, critic_loss_val) # Return individual losses as well


    # Compute gradients for the combined loss, and also get individual losses
    (combined_loss, auxiliary_losses), grads = jax.value_and_grad(combined_loss_fn, has_aux=True)([actor_params, critic_params]) # has_aux=True to get auxiliary output
    actor_loss_val, critic_loss_val = auxiliary_losses # Unpack auxiliary losses
    actor_grads, critic_grads = grads

    # Apply gradients using optimizers
    actor_updates, actor_opt_state   = actor_optimizer.update(actor_grads, actor_opt_state, actor_params)
    critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state, critic_params)

    actor_params  = optax.apply_updates(actor_params, actor_updates)
    critic_params = optax.apply_updates(critic_params, critic_updates)

    return actor_params, actor_opt_state, critic_params, critic_opt_state, combined_loss, actor_loss_val, critic_loss_val # Return individual losses for logging


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
        episode_rewards    = 0
        episode_sentences  = []
        episode_states, episode_actions, episode_rewards_list, episode_log_probs, episode_values = [], [], [], [], []

        sentence_tokens    = []
        current_state      = 0 # Start with initial state (e.g., index 0 or <BOS> if you have one)

        for step in range(sentence_length):
            rng, action_key, value_key = jax.random.split(rng, 3)

            # Actor inference for action probabilities
            action_probabilities = actor_inference(actor_params, jnp.asarray([current_state])) # Pass current state

            action_probabilities = np.array(action_probabilities[0]) # Get probabilities for current state
            action_index         = np.random.choice(vocab_size, p=action_probabilities)

            # Critic inference for value estimation
            predicted_value      = critic_inference(critic_params, jnp.asarray([current_state])) # Pass current state
            value                = np.array(predicted_value[0]).item() # Extract scalar value
            log_prob             = jnp.log(action_probabilities[action_index])

            action = int(action_index)

            episode_states.append(current_state) # Append current state
            episode_actions.append(action)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_rewards_list.append(0) # Dummy reward, will be calculated later

            sentence_tokens.append(action)
            current_state = action # Update current state for next step

            if action_index == eos_token_index:
                break


        sentence_text   = decode_text(sentence_tokens)
        reward          = reward_model(sentence_text)
        episode_rewards = reward

        advantages = gae_advantage(episode_rewards_list, episode_values, 0, gamma, gae_lambda)
        returns    = advantages + np.array(episode_values)

        # Create batch data
        batch_data = (
            np.array(episode_states   , dtype=np.int32  ), # States as integer array
            np.array(episode_actions  , dtype=np.int32  ), # Actions as integer array
            np.array(episode_log_probs, dtype=np.float32),
            advantages.astype(np.float32)                , # Ensure advantages are float32
            returns.astype(np.float32)                     # Ensure returns are float32
        )

        # PPO Policy Update Loop
        for _ in range(policy_epochs):
            perm = np.random.permutation(len(episode_states)) # Corrected permutation here
            for start_idx in range(0, len(episode_states), mini_batch_size):
                mini_batch_idx = perm[start_idx:start_idx + mini_batch_size]


                mini_batch = tuple(arr[mini_batch_idx] for arr in batch_data)
                actor_params, actor_opt_state, critic_params, critic_opt_state, combined_loss, actor_loss, critic_loss = train_step( # Get individual losses here
                    actor_params, actor_opt_state, critic_params, critic_opt_state, mini_batch
                )


        print(f"Episode {episode}, Reward: {episode_rewards:.2f}, Sentence: '{sentence_text}', Combined Loss: {combined_loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}") # Print individual losses

finally:
    pass