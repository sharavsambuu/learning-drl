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
# - Prompt Masking: Асуултыг (Prompt) сургалтад оруулахгүй, зөвхөн 
#   моделийн өөрийнх нь үүсгэсэн хариулт (Completion) дээр сургана
# - Stable Advantage: Бүлэг доторх оноог харьцуулахдаа STD-д хуваахгүй
#   Mean Centering + Clipping хийж тогтворжуулна
# - KL Divergence тооцохдоо padding болон prompt хэсгийг хасна
# - Gradient Accumulation: VRAM хэмнэх үүднээс 1, 1-ээр нь цувуулж сургана
#
# Тохиргоо:
# Hardware : RTX 5070 Ti (12GB VRAM)
# Precision: bfloat16 (Санах ойг 2 дахин хэмнэнэ)
#

import os
# JAX санах ойн хуваарилалтыг хязгаарлах буюу VRAM дүүрэхээс сэргийлнэ
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

import re
import random
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.training import train_state
from transformers import AutoTokenizer, FlaxLlamaForCausalLM
from datasets import load_dataset

# RTX GPU дээр тооцооллыг хурдасгах
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

seed = 42
np.random.seed(seed)
random.seed(seed)
# JAX Random Key-ийг глобал хувьсагчаар биш, loop дотор дамжуулж ашиглана
init_key = jax.random.PRNGKey(seed)


# HYPERPARAMETERS (12GB VRAM SAFE)

# Модель болон Датасет
model_id            = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
max_ctx_len         = 2048      # Моделийн тохиргооноос дараа нь шинэчилнэ
max_gen_len         = 512       # Моделийн "бодох" дээд хязгаар

# GRPO Сургалтын тохиргоо
total_updates       = 200       # Демо учраас цөөхөн алхам сургах
group_size          = 4         # Нэг асуултыг 4 янзаар бодуулж өрсөлдүүлнэ
mini_batch_size     = 1         # GPU дээр нэг удаад нэг л prompt оруулна
learning_rate       = 2e-6      # Мэдлэгээ мартахгүйгээр бага багаар сайжрах хурд
ppo_epochs          = 2         # Нэг rollout дээр хэдэн удаа давтаж сургах вэ
clip_epsilon        = 0.2       # PPO clipping range (Хэт огцом өөрчлөлтөөс сэргийлнэ)
kl_beta             = 0.04      # Хуучин моделиосоо хэт хол зөрөхгүй байх тохиргоо
adv_clip_range      = 5.0       # Advantage-ийг хэт савлуулахгүйн тулд хязгаарлана

# System Prompt
system_prompt_text = (
    "You are a reasoning expert.\n"
    "1. You MUST enclose your thought process inside <think> ... </think> tags.\n"
    "2. You MUST put your final numeric answer inside <answer> ... </answer> tags.\n"
    "3. Think step-by-step."
)

print("="*50)
print("  MINIMUM VIABLE REASONER: GRPO TRAINING")
print(f"  Model: {model_id}")
print(f"  Updates: {total_updates} | Group: {group_size}")
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
    return messages


# MODEL & TOKENIZER SETUP

print("Tokenizer болон Model ачаалж байна...")
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

# Моделийн өөрийнх нь тохиргооноос Context Window-г уншиж авах
# Хэрэв олдохгүй бол анхны тохируулсан 2048-аар явна
try:
    max_ctx_len = int(getattr(model.config, "max_position_embeddings", max_ctx_len))
except Exception:
    pass

# Reference Model (Хөлдөөсөн хувилбар)
# Санах ой хэмнэхийн тулд жинхэнэ copy үүсгэхгүй
# JAX-д params_ref = model.params гэхэд санах ойд шинээр зай эзлэхгүй зөвхөн pointer заана 
# Сургалтын явцад train_state.params өөрчлөгдөхөд params_ref хуучнаараа үлдэнэ
params     = model.params
params_ref = model.params 

print(f"Модель амжилттай ачаалагдлаа. (Context: {max_ctx_len})")


# JAX HELPERS

def build_attn_mask_from_eos(seqs, eos_id, prompt_len):
    """
    EOS token болон PAD token ижилхэн ID-тай байх үед (SmolLM/Llama)
    энгийн (seq != pad) mask ашиглавал өгүүлбэр дундах EOS-ийг mask хийх эрсдэлтэй.
    Тиймээс Prompt-ийн дараа гарч ирсэн АНХНЫ EOS token хүртэл л хүчинтэйд тооцно.
    """
    # seqs: [Batch, Time]
    T            = seqs.shape[1]
    # Prompt хэсгийг алгасаад хайх
    after_prompt = (jnp.arange(T)[None, :] >= prompt_len).astype(jnp.int32)
    eos_hits     = (seqs == eos_id).astype(jnp.int32) * after_prompt
    
    # Хамгийн эхний EOS-ийн байрлалыг олох
    first_eos    = jnp.argmax(eos_hits, axis=1)
    has_eos      = jnp.any(eos_hits == 1, axis=1)
    
    # EOS байхгүй бол дуустал, байвал EOS-ийг оролцуулаад таслах
    end_idx      = jnp.where(has_eos, first_eos + 1, T)
    
    # Mask үүсгэх [B, T]
    mask         = (jnp.arange(T)[None, :] < end_idx[:, None]).astype(jnp.int32)
    return mask


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
    GRPO-ийн гол арга, текстийг уншиж оноо өгөх хэсэг.
    Урт текстэд өгдөг байсан бонусыг багасгаж, зөв хариултад илүү төвлөрнө.
    """
    rewards  = []
    true_val = extract_ground_truth(ground_truth_text)
    
    for text in rollouts:
        score = 0.0
        
        # FORMAT REWARDS (XML бүтцээ зөв бичсэн эсэх)
        has_think  = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text
        
        if has_think:
            score += 0.1
        
        if has_answer:
            # Хариултын таг дотор тоо, цэг, таслал, $, зайгаас өөр "хог" байвал шийтгэнэ
            # Энэ нь моделийг зөвхөн тоо бичдэг болгоход тусална (Strict Parsing)
            ans_content = text.split("<answer>")[1].split("</answer>")[0].strip()
            if re.search(r'[^0-9\-\.\,\$\s]', ans_content):
                score -= 0.5
            else:
                score += 0.2
                
            pred_val = extract_xml_answer(text)
            
            # CORRECTNESS REWARD (Хариу зөв үү?)
            if pred_val is not None and true_val is not None:
                if abs(pred_val - true_val) < 1e-4:
                    score += 2.0
                    # Зөв хариулсан тохиолдолд л reasoning урт байвал урамшуулна
                    # Гэхдээ <think> байгаа эсэхийг заавал шалгана (Crash-аас сэргийлнэ)
                    if has_think:
                        thought = text.split("<think>")[1].split("</think>")[0]
                        if len(thought) > 100: 
                            score += 0.2
                else:
                    score -= 0.5
        else:
            # Хариултын tag байхгүй бол формат буруу гэж үзээд хатуу шийтгэнэ
            score -= 1.0

        rewards.append(score)
        
    return np.array(rewards)

def compute_advantages_stable(rewards):
    """ 
    Group доторх дундажтай харьцуулж Advantage тооцно.
    Жижиг group size үед STD-д хуваах нь эрсдэлтэй тул
    зөвхөн Mean Centering хийгээд Clipping хийнэ.
    """
    mean       = np.mean(rewards)
    advantages = rewards - mean
    # Хэт өндөр утгыг хайчилна (Stable Training)
    advantages = np.clip(advantages, -adv_clip_range, adv_clip_range)
    return advantages


# TRAINING STATE & OPTIMIZER

# Adafactor optimizer болон MultiSteps ашиглан VRAM хэмнэнэ
# every_k_schedule=group_size гэдэг нь group_size удаа алхам хийсний дараа 
# сая нэг удаа жингээ шинэчилнэ гэсэн үг (Gradient Accumulation)
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
def train_step(state, params_ref, input_ids, attention_mask, advantages, old_log_probs, prompt_len):
    """
    GRPO Update Step (PPO Logic)
    Prompt Masking болон KL calculation агуулсан
    """
    def loss_fn(p):
        outputs = state.apply_fn(input_ids=input_ids, attention_mask=attention_mask, params=p)
        logits  = outputs.logits
        all_lps = jax.nn.log_softmax(logits, axis=-1)
        
        # Action Log Probs (Shifted right)
        # input[i] нь input[i+1]-ийг таамаглана
        act_lps = jnp.take_along_axis(
            all_lps[:, :-1, :], 
            input_ids[:, 1:, None], 
            axis=-1
        ).squeeze(-1)
        
        # Prompt Masking
        # Зөвхөн prompt-оос хойшхи token-ууд дээр сургана
        # Асуултыг цээжлэх биш, хариулт дээр сайжрах ёстой
        seq_len   = input_ids.shape[1]
        pos_idxs  = jnp.arange(seq_len - 1)[None, :] 
        gen_mask  = (pos_idxs >= (prompt_len - 1)).astype(jnp.int32)
        
        # Padding mask + Gen mask
        # attention_mask[:, 1:] нь padding-ийг хасна
        loss_mask = attention_mask[:, 1:] * gen_mask
        
        # PPO Ratio Calculation
        ratio = jnp.exp(act_lps - old_log_probs)
        
        # Surrogate Loss - PPO-ийн гол томъёо
        adv_broad = advantages[:, None] 
        surr1     = ratio * adv_broad
        surr2     = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_broad
        ppo_loss  = -jnp.minimum(surr1, surr2)
        
        # KL Divergence (Зөвхөн generated tokens дээр)
        ref_out   = state.apply_fn(input_ids=input_ids, attention_mask=attention_mask, params=params_ref)
        ref_lps   = jax.nn.log_softmax(ref_out.logits, axis=-1)
        
        kl_full   = jnp.sum(jnp.exp(all_lps) * (all_lps - ref_lps), axis=-1)
        kl_gen    = kl_full[:, :-1] 
        
        # Total Loss
        # Mask ашиглан дунджийг зөв тооцох (Sum / Count)
        valid_toks = jnp.sum(loss_mask) + 1e-8
        final_loss = jnp.sum((ppo_loss + kl_beta * kl_gen) * loss_mask) / valid_toks
        
        # Stats for logging
        avg_kl     = jnp.sum(kl_gen * loss_mask) / valid_toks
        avg_ratio  = jnp.sum(ratio * loss_mask) / valid_toks
        
        return final_loss, (avg_kl, avg_ratio)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (m_kl, m_ratio)), grads = grad_fn(state.params)
    
    # MultiSteps optimizer ашиглаж байгаа тул apply_gradients нь
    # градиентыг цуглуулж байгаад k алхмын дараа жинг шинэчилнэ
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, m_kl, m_ratio

def generate_rollouts_batched(state, prompt_ids, prompt_mask, key, max_new):
    """
    Rollout generation хийхдээ explicit PRNG key болон уртын хязгаарыг ашиглана.
    Мөн EOS token-ийг зааж өгснөөр модель дуусах цэгээ мэднэ.
    """
    batch_input = jnp.repeat(prompt_ids, group_size, axis=0)
    
    outputs = model.generate(
        batch_input,
        params         = state.params,
        max_new_tokens = max_new,
        do_sample      = True,
        temperature    = 0.8,
        top_k          = 50,
        pad_token_id   = tokenizer.pad_token_id,
        eos_token_id   = tokenizer.eos_token_id, # Generation зогсох нөхцөл
        prng_key       = key 
    )
    return outputs.sequences


# MAIN TRAINING LOOP

print("\n" + "="*50)
print("  PHASE 2: GRPO LOOP (ACCUMULATED + MASKED)")
print(f"  Updates: {total_updates} | Group: {group_size} | Epochs: {ppo_epochs}")
print("="*50 + "\n")

step_counter = 0
curr_key     = init_key

while step_counter < total_updates:
    # Датасетээс санамсаргүй нэг жишээ авах
    idx              = random.randint(0, len(dataset) - 1)
    example          = dataset[idx]
    question         = example['question']
    ground_truth_raw = example['answer'  ]
    
    # Prompt бэлтгэх
    messages   = format_gsm8k_prompt(question)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>"
    inputs     = tokenizer(input_text, return_tensors="np", max_length=max_ctx_len, truncation=True)
    
    prompt_ids = jnp.array(inputs['input_ids'])
    prompt_len = prompt_ids.shape[1]
    
    # Context цонх шалгалт
    # Prompt хэт урт бол generation хийх зай үлдэхгүй тул алгасна
    if prompt_len >= max_ctx_len - 50:
        continue
        
    # Safe max tokens calculation (Context limit-ээс хэтрэхгүй байх)
    # 0 болон хасах утга гарахаас сэргийлж дор хаяж 1 token үүсгэнэ
    safe_max_new = max(1, min(max_gen_len, max_ctx_len - prompt_len))
    
    # Rollout үе шат
    curr_key, gen_key = jax.random.split(curr_key)
    try:
        sequences = generate_rollouts_batched(train_state_obj, prompt_ids, None, gen_key, safe_max_new)
    except Exception as e:
        print(f"[Gen Error] {e}")
        jax.clear_caches()
        continue

    # Reward & Advantage тооцох
    decoded_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    rewards       = compute_rewards(decoded_texts, ground_truth_raw)
    advantages    = compute_advantages_stable(rewards) 
    
    # Old Log Probs (Training step дотор биш гадна талд бүтнээр нь авна)
    # Энгийн (seq != pad) биш, EOS-ийг зөв тооцох тусгай mask ашиглана
    attn_mask = build_attn_mask_from_eos(sequences, tokenizer.eos_token_id, prompt_len)
    
    def get_log_probs(p, ids, mask):
        logits = model(input_ids=ids, attention_mask=mask, params=p).logits
        lps    = jax.nn.log_softmax(logits, axis=-1)
        # Shift хийгдсэн (сүүлийн token хасагдсан) утга буцаана
        return jnp.take_along_axis(lps[:, :-1, :], ids[:, 1:, None], axis=-1).squeeze(-1)

    try:
        old_log_probs = get_log_probs(train_state_obj.params, sequences, attn_mask)
    except Exception as e:
         print(f"[Ref Error] {e}")
         jax.clear_caches()
         continue

    # Training Step - Gradient Accumulation Loop
    adv_jax     = jnp.array(advantages)
    batch_loss  = 0
    batch_kl    = 0
    batch_ratio = 0
    
    try:
        # PPO Epochs: Нэг rollout дээр олон удаа давтаж сургах нь үр дүнтэй
        # Gradient accumulation учир epoch бүрт optimizer update хийгдэнэ
        for _ in range(ppo_epochs):
            for i in range(group_size):
                sub_seq  = sequences    [i:i+1]
                sub_mask = attn_mask    [i:i+1]
                sub_adv  = adv_jax      [i:i+1]
                sub_old  = old_log_probs[i:i+1]
                
                # Prompt_len-ийг дамжуулж зөв masking хийнэ
                train_state_obj, loss_val, kl_val, ratio_val = train_step(
                    train_state_obj, 
                    params_ref     , 
                    sub_seq        , 
                    sub_mask       , 
                    sub_adv        , 
                    sub_old        ,
                    prompt_len
                )
                batch_loss  += loss_val
                batch_kl    += kl_val
                batch_ratio += ratio_val
            
        step_counter += 1
        
        # Явцыг хэвлэж харах
        if step_counter % 5 == 0:
            avg_r     = np.mean(rewards)
            divisor   = group_size * ppo_epochs
            avg_loss  = batch_loss  / divisor
            avg_kl_v  = batch_kl    / divisor
            avg_ratio = batch_ratio / divisor
            
            print(f"[Step {step_counter:3d}] Reward: {avg_r:+.2f} | Loss: {avg_loss:.4f} | KL: {avg_kl_v:.4f} | Ratio: {avg_ratio:.4f}")
            
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
        print(f"[Train Error] {e}")
        jax.clear_caches()
        continue

print("\n=== СУРГАЛТ АМЖИЛТТАЙ ДУУСЛАА ===")

# Төгсгөлийн шалгалт
test_q   = "If I have 5 apples and eat 2, how many do I have?"
test_msg = format_gsm8k_prompt(test_q)
test_in  = tokenizer.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True) + "<think>"
test_tok = tokenizer(test_in, return_tensors="np")

final_gen = model.generate(
    jnp.array(test_tok['input_ids']),
    params         = train_state_obj.params,
    max_new_tokens = 200,
    do_sample      = False,
    eos_token_id   = tokenizer.eos_token_id
)
print("\nFINAL TEST ANSWER:")
print(tokenizer.decode(final_gen.sequences[0], skip_special_tokens=True))