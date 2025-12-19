#
# Flax friendly Small Language Model inference 
#
#


import os
import jax
import jax.numpy    as jnp
from   transformers import AutoTokenizer, FlaxLlamaForCausalLM

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

def run_inference():
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    print(f"Loading {model_id} with Flax...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"\n[ERROR] Could not load tokenizer: {e}")
        return

    try:
        # Access params : model.params
        # Forward pass  : model(input_ids, params=params).logits
        model = FlaxLlamaForCausalLM.from_pretrained(
            model_id               ,
            from_pt  = True        ,    # PyTorch weights
            dtype    = jnp.bfloat16,    # Efficient for 5070 Ti
        )
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        print("Maybe: pip install torch")
        return

    print("Model loaded successfully!")
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful math assistant. Think step by step."
        },
        {
            "role": "user", 
            "content": "A store sells apples for $2 each and oranges for $3 each. If I buy 4 apples and 3 oranges, how much do I pay?"
        }
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages                     , 
        tokenize              = False, 
        add_generation_prompt = True
    )
    
    inputs = tokenizer(text_input, return_tensors="np")
    print(f"\nInput: {text_input}\n")

    key = jax.random.PRNGKey(42)
    print("Thinking...")
    
    outputs = model.generate(
        inputs["input_ids"],
        prng_key       = key, 
        max_new_tokens = 128,
        do_sample      = True,
        temperature    = 0.6,
        pad_token_id   = tokenizer.eos_token_id
    )

    input_len    = inputs["input_ids"].shape[1]
    response_ids = outputs.sequences[0][input_len:]
    response     = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    print("-" * 30)
    print(f"OUTPUT:\n{response}")
    print("-" * 30)

if __name__ == "__main__":
    print(f"JAX Devices: {jax.devices()}")
    run_inference()