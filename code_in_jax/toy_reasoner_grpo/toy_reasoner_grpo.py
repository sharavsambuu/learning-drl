#
# MINIMUM VIABLE REASONER, GSM8K MATH SOLVER WITH GRPO
#
# Зорилго:
# Энэхүү код нь SmolLM2-1.7B-Instruct хэмээх жижигхэн LLM-ийг 
# GRPO (Group Relative Policy Optimization) аргаар сайжруулж, 
# GSM8K датасет дээрх бодлогуудыг зүгээр нэг хариулах биш яг 
# хүн шиг задалж боддог (Reasoning) болгож сургах явдал юм.
#
# Машин сургалтын арга барил:
# - Нэг бодлогыг модельд өгөөд олон янзаар бодуулна (Rollouts)
# - Хариулт бүрийг <think>, <answer> tag ашигласан байдал болон
#   эцсийн хариу зөв эсэхээр нь дүгнэнэ (Reward Function)
# - Gradient Accumulation ашиглан VRAM хэмнэж, бүлэг дотроо 
#   хамгийн сайн байсанд нь урам өгч (Advantage) сургана
#
# Тохиргоо:
# Hardware : RTX 5070 Ti (12GB VRAM-д тааруулсан тохиргоо)
# Precision: bfloat16 (Санах ойг 2 дахин хэмнэнэ)
#

import os
# JAX санах ойн хуваарилалтыг хязгаарлах (VRAM дүүрэхээс сэргийлнэ)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

import re
import time
import random
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import AutoTokenizer, FlaxLlamaForCausalLM
from datasets import load_dataset

# RTX GPU дээр тооцооллыг хурдасгах
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

seed = 42
np.random.seed(seed)
random.seed(seed)
key = jax.random.PRNGKey(seed)


# HYPERPARAMETERS (12GB VRAM SAFE)

# Модель болон Датасет
model_id            = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
max_seq_len         = 1024      # Accumulation ашиглаж байгаа тул урт байж болно
max_gen_len         = 512       # Моделийн "бодох" дээд хязгаар

# GRPO Сургалтын тохиргоо
total_updates       = 200       # Демо учраас цөөхөн алхам сургах
group_size          = 4         # Нэг асуултыг 4 янзаар бодуулж өрсөлдүүлнэ
mini_batch_size     = 1         # GPU дээр нэг удаад нэг л prompt оруулна
learning_rate       = 2e-6      # Мэдлэгээ мартахгүйгээр бага багаар сайжрах хурд
ppo_epochs          = 1         # Нэг rollout дээр хэдэн удаа давтаж сургах вэ
clip_epsilon        = 0.2       # PPO clipping range (Хэт огцом өөрчлөлтөөс сэргийлнэ)
kl_beta             = 0.04      # Хуучин моделиосоо хэт хол зөрөхгүй байх тохиргоо

# System Prompt
system_prompt_text = (
    "You are a reasoning expert.\n"
    "1. You MUST enclose your thought process inside <think> ... </think> tags.\n"
    "2. You MUST put your final numeric answer inside <answer> ... </answer> tags.\n"
    "3. Think step-by-step."
)

print("="*50)
print("  TINY REASONER: GRPO + ACCUMULATION TRAINING START")
print(f"  Model: {model_id}")
print(f"  Device: {jax.devices()[0]}")
print("="*50 + "\n")


# DATASET LOADING & PREPARATION

# GSM8K буюу Grade School Math датасетийг татах
try:
    dataset = load_dataset("gsm8k", "main", split="train")
    print(f"Dataset loaded: {len(dataset)} жишээ бэлэн байна.")
except Exception as e:
    print(f"Dataset error: {e}")
    exit()

def format_gsm8k_prompt(question):
    """
    SmolLM2-ийн chat template-ийг ашиглан prompt бэлтгэх.
    System prompt-ийг оруулж өгснөөр модель дүрдээ орно.
    """
    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user"  , "content": question          }
    ]
    # Tokenizer-ийг дараа нь ашиглах тул энд түүхий текст буцаана
    return messages


# MODEL & TOKENIZER SETUP

print("[1/3] Tokenizer болон Model ачаалж байна...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Padding token байхгүй бол EOS-ийг ашиглана (Generation хийхэд хэрэгтэй)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Моделийг bfloat16 форматаар ачаалах (VRAM хэмнэх гол арга)
model = FlaxLlamaForCausalLM.from_pretrained(
    model_id,
    from_pt=True,       # PyTorch жингээс хөрвүүлж байна
    dtype=jnp.bfloat16, # GPU санах ойг 2 дахин хэмнэнэ
)

# Reference Model (Хөлдөөсөн хувилбар)
# Санах ой хэмнэхийн тулд жинхэнэ copy үүсгэхгүй
# JAX-д params_ref = model.params гэхэд санах ойд шинээр зай эзлэхгүй зөвхөн pointer заана 
# Сургалтын явцад train_state.params өөрчлөгдөхөд params_ref хуучнаараа үлдэнэ
# Copy-on-write шиг ажиллана
params     = model.params
params_ref = model.params 

print("[2/3] Модель амжилттай ачаалагдлаа. (Reasoning хийхэд бэлэн)")


# REWARD FUNCTIONS (дүн тавьдаг багш)

def extract_xml_answer(text):
    """ <answer>123</answer> дотроос тоог нь сугалж авах """
    try:
        if "<answer>" in text and "</answer>" in text:
            ans_part = text.split("<answer>")[1].split("</answer>")[0]
            # Тоо, цэг, хасах тэмдгийг үлдээгээд бусдыг цэвэрлэх
            number = re.findall(r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?', ans_part)
            if number:
                return float(number[-1].replace(',', '').replace('$', ''))
    except:
        pass
    return None

def extract_ground_truth(text):
    """ GSM8K-ийн '#### 42' форматаас зөв хариуг авах """
    if "####" in text:
        ans_part = text.split("####")[-1].strip()
        number = re.findall(r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?', ans_part)
        if number:
            return float(number[-1].replace(',', '').replace('$', ''))
    return None

def compute_rewards(rollouts, ground_truth_text):
    """
    GRPO-ийн гол арга, текстийг уншиж оноо өгөх хэсэг
    """
    rewards  = []
    true_val = extract_ground_truth(ground_truth_text)
    
    for text in rollouts:
        score = 0.0
        
        # FORMAT REWARDS (XML бүтцээ зөв бичсэн эсэх)

        # <think> tag байгаа эсэх, бодож эхэлсэн үү?
        if "<think>" in text:
            score += 0.2
            if "</think>" in text:
                score += 0.2
                # Бодолт хэт богино биш байх ёстой, худлаа таагаагүй байх ёстой
                thought_content = text.split("<think>")[1].split("</think>")[0]
                if len(thought_content) > 100:
                    score += 0.1
        
        # <answer> tag байгаа эсэх
        if "<answer>" in text and "</answer>" in text:
            score += 0.3
            
            # CORRECTNESS REWARD, хариу зөв үү
            pred_val = extract_xml_answer(text)
            if pred_val is not None and true_val is not None:
                # Маш жижиг зөрүүг (floating point error) зөвшөөрнө
                if abs(pred_val - true_val) < 1e-4:
                    score += 2.0  # Зөв бол том шагнал
                else:
                    score -= 0.5  # Буруу бол шийтгэл
        else:
            # Хариултын tag байхгүй бол формат буруу гэж үзээд хатуу шийтгэнэ
            score -= 1.0

        rewards.append(score)
        
    return np.array(rewards)

def compute_advantages(rewards):
    """ Group доторх дундажтай харьцуулж хэн нь 'онц', хэн нь 'муу' сурсныг тооцно """
    mean = np.mean(rewards)
    std  = np.std(rewards) + 1e-8
    # Normalize rewards 
    advantages = (rewards - mean) / std
    return advantages


# TRAINING STATE & STEP FUNCTION (GRADIENT ACCUMULATION)

# Adafactor optimizer болон MultiSteps ашиглан VRAM хэмнэнэ
# every_k_schedule=group_size гэдэг нь group_size удаа алхам хийсний дараа 
# сая нэг удаа жингээ шинэчилнэ гэсэн үг (Accumulation)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adafactor(learning_rate=learning_rate)
)
optimizer = optax.MultiSteps(optimizer, every_k_schedule=group_size)

train_state_obj = train_state.TrainState.create(
    apply_fn = model.__call__,
    params   = params        ,
    tx       = optimizer
)

@jax.jit
def train_step(state, params_ref, input_ids, attention_mask, advantages, old_log_probs):
    """
    GRPO Update Step (PPO Logic)
    Accumulation хийж байгаа тул энэ функц жижиг batch (1 ширхэг) дээр ажиллана
    """
    def loss_fn(p):
        # Одоогийн моделийн Logits тооцох
        outputs = state.apply_fn(input_ids=input_ids, attention_mask=attention_mask, params=p)
        logits  = outputs.logits
        
        # Сүүлийн token-оос бусад бүх token-ий log_softmax авах
        # (Shift хийх шаардлагатай: input[i] нь input[i+1]-ийг таамаглана)
        all_log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Target Log Probs (яг сонгосон token-уудын магадлал)
        # input_ids shape: [B, SeqLen] -> action_log_probs shape: [B, SeqLen-1]
        action_log_probs = jnp.take_along_axis(
            all_log_probs[:, :-1, :], 
            input_ids[:, 1:, None], 
            axis=-1
        ).squeeze(-1)
        
        # Masking (Padding хэсгийг тооцохгүй, зөвхөн жинхэнэ текст дээр сурна)
        loss_mask = attention_mask[:, 1:]
        
        # PPO Ratio Calculation
        # old_log_probs нь гадна талд аль хэдийн shift хийгдсэн тул шууд ашиглана
        ratio = jnp.exp(action_log_probs - old_log_probs)
        
        # Surrogate Loss - PPO-ийн гол томъёо
        adv_broadcast = advantages[:, None] 
        surr1         = ratio * adv_broadcast
        surr2         = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_broadcast
        ppo_loss      = -jnp.minimum(surr1, surr2)
        
        # KL Divergence Penalty (Хэт галзуурахаас сэргийлнэ)
        ref_outputs   = state.apply_fn(input_ids=input_ids, attention_mask=attention_mask, params=params_ref)
        ref_logits    = ref_outputs.logits
        ref_log_probs = jax.nn.log_softmax(ref_logits, axis=-1)
        
        kl_div = jnp.exp(all_log_probs) * (all_log_probs - ref_log_probs)
        kl_div = jnp.sum(kl_div, axis=-1) # Sum over vocab
        
        # Total Loss = PPO Loss + KL Penalty
        total_loss = jnp.sum((ppo_loss + kl_beta * kl_div[:, :-1]) * loss_mask) / jnp.sum(loss_mask)
        
        return total_loss, (jnp.mean(kl_div), jnp.mean(ratio))

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (mean_kl, mean_ratio)), grads = grad_fn(state.params)
    
    # MultiSteps optimizer ашиглаж байгаа тул apply_gradients нь
    # градиентыг цуглуулж байгаад k алхмын дараа жинг шинэчилнэ
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss, mean_kl

# Generate функц (Олон хувилбараар бодуулж үзэх)
def generate_rollouts_batched(state, prompt_ids, prompt_mask):
    # Prompt-ийг group_size удаа хувилна
    # Inference дээр VRAM ачаалал бага тул batch-аар хийж болно
    # Хэрэв энд OOM өгвөл loop-д оруулж болно
    batch_input = jnp.repeat(prompt_ids, group_size, axis=0)
    
    # Flax generate ашиглах
    outputs = model.generate(
        batch_input,
        params         = state.params          ,
        max_new_tokens = max_gen_len           ,
        do_sample      = True                  ,
        temperature    = 0.8                   ,  # Reasoning хийхэд хэт бага биш, хэт их биш temp зүгээр
        top_k          = 50                    ,
        pad_token_id   = tokenizer.pad_token_id,
        prng_key       = jax.random.PRNGKey(int(time.time()))
    )
    
    return outputs.sequences


# MAIN TRAINING LOOP

print("\n" + "="*50)
print("  PHASE 2: GRPO LOOP - Reasoner болгох сургалт")
print(f"  Updates: {total_updates} | Group Size: {group_size}")
print("="*50 + "\n")

step_counter = 0

while step_counter < total_updates:
    # Датасетээс санамсаргүй нэг жишээ авах
    idx              = random.randint(0, len(dataset) - 1)
    example          = dataset[idx]
    question         = example['question']
    ground_truth_raw = example['answer'  ]
    
    # Prompt бэлтгэх
    messages   = format_gsm8k_prompt(question)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_text += "<think>"
    
    inputs      = tokenizer(input_text, return_tensors="np", max_length=max_seq_len, truncation=True)
    prompt_ids  = jnp.array(inputs['input_ids'     ])
    prompt_mask = jnp.array(inputs['attention_mask'])
    
    # Rollout үе шат (Олон хувилбараар бодуулах)
    try:
        sequences = generate_rollouts_batched(train_state_obj, prompt_ids, prompt_mask)
    except Exception as e:
        print(f"[Warning] Generation failed (OOM?): {e}")
        jax.clear_caches()
        continue

    # Reward тооцоолох (Багшийн үнэлгээ)
    decoded_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    rewards       = compute_rewards(decoded_texts, ground_truth_raw)
    advantages    = compute_advantages(rewards)
    
    # Old Log Probs (No Grad) - Хуучин мэдлэгтэйгээ харьцуулахын тулд
    attn_mask = (sequences != tokenizer.pad_token_id).astype(jnp.int32)
    def get_log_probs(p, ids, mask):
        logits    = model(input_ids=ids, attention_mask=mask, params=p).logits
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Shift хийгдсэн (сүүлийн token хасагдсан) утга буцаана
        return jnp.take_along_axis(log_probs[:, :-1, :], ids[:, 1:, None], axis=-1).squeeze(-1)

    try:
        old_log_probs = get_log_probs(train_state_obj.params, sequences, attn_mask)
    except Exception as e:
        print(f"[Ref Error] {e}")
        jax.clear_caches()
        continue

    # Train Step - Gradient Accumulation Loop
    # VRAM хэмнэхийн тулд 4 жишээг 1, 1-ээр нь цувуулж гүйлгэнэ
    adv_jax    = jnp.array(advantages)
    batch_loss = 0
    batch_kl   = 0
    
    try:
        for i in range(group_size):
            # i-р жишээг сугалж авах (Shape: [1, SeqLen])
            # Энэ нь VRAM-д маш бага ачаалал өгнө
            sub_seq  = sequences    [i:i+1]
            sub_mask = attn_mask    [i:i+1]
            sub_adv  = adv_jax      [i:i+1]
            sub_old  = old_log_probs[i:i+1]
            
            # Энэ функц group_size удаа дуудагдана 
            # MultiSteps optimizer үүнийг автоматаар цуглуулж байгаад 
            # хамгийн сүүлийн алхам дээр жингээ шинэчилнэ
            train_state_obj, loss_val, kl_val = train_step(
                train_state_obj, 
                params_ref     , 
                sub_seq        , 
                sub_mask       , 
                sub_adv        , 
                sub_old
            )
            batch_loss += loss_val
            batch_kl   += kl_val
            
        step_counter += 1
        
        # Явцыг хэвлэж харах
        if step_counter % 5 == 0:
            avg_r = np.mean(rewards)
            print(f"[Step {step_counter:3d}] Reward: {avg_r:+.2f} | Loss: {batch_loss/group_size:.4f} | KL: {batch_kl/group_size:.4f}")
            
        if step_counter % 20 == 0:
            best_idx = np.argmax(rewards)
            print("\n" + "-"*30)
            print(f"QUESTION: {question[:100]}...")
            print(f"BEST SAMPLE (R={rewards[best_idx]:.1f}):")
            output_sample = decoded_texts[best_idx]
            if "<think>" in output_sample:
                print(output_sample.split("<think>", 1)[1][:300] + "...")
            else:
                print(output_sample[:200] + "...")
            print("-"*30 + "\n")
            
    except Exception as e:
        print(f"[Error] Update step failed: {e}")
        jax.clear_caches()
        continue

print("\n=== СУРГАЛТ АМЖИЛТТАЙ ДУУСЛАА ===")
print("Одоо модель <think> tag ашиглан боддог болсон байх ёстой.")

# Төгсгөлийн шалгалт
test_q   = "If I have 5 apples and eat 2, how many do I have?"
test_msg = format_gsm8k_prompt(test_q)
test_in  = tokenizer.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True) + "<think>"
test_tok = tokenizer(test_in, return_tensors="np")

final_gen = model.generate(
    jnp.array(test_tok['input_ids']),
    params         = train_state_obj.params,
    max_new_tokens = 200,
    do_sample      = False  # Шалгалтын үед санамсаргүй байдлыг унтраана (Greedy)
)
print("\nFINAL TEST ANSWER:")
print(tokenizer.decode(final_gen.sequences[0], skip_special_tokens=True))