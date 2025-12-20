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
# - Fixed Shapes: JAX JIT recompilation-аас сэргийлэхийн тулд
#   бүх prompt-ийг тогтмол урттай болгож pad хийнэ
# - Gradient Accumulation: VRAM хэмнэх үүднээс 1, 1-ээр нь цувуулж сургана
#
# Тохиргоо:
# Hardware : 10GB VRAM Optimization
# Precision: bfloat16 (Санах ойг 2 дахин хэмнэнэ)
#

import os
# JAX санах ойн хуваарилалтыг хязгаарлах буюу VRAM дүүрэхээс сэргийлнэ
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"  ] = "platform"

import re
import random
import numpy         as np
import jax
import jax.numpy     as jnp
import optax
from   flax          import jax_utils
from   flax.training import train_state
from   transformers  import AutoTokenizer, FlaxLlamaForCausalLM
from   datasets      import load_dataset

# RTX GPU дээр тооцооллыг хурдасгах
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

seed = 42
np.random.seed(seed)
random.seed(seed)
# JAX Random Key-ийг глобал хувьсагчаар биш, loop дотор дамжуулж ашиглана
init_key = jax.random.PRNGKey(seed)


# HYPERPARAMETERS (10GB VRAM OPTIMIZED)

# Модель болон Датасет
model_id            = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
max_ctx_len         = 1536      # 10GB VRAM-д багтаах context урт
max_gen_len         = 256       # Reasoning хэт урт байвал VRAM дүүрэх эрсдэлтэй
# Fixed Shape буюу JAX JIT-ийг дахин ачааллуулахгүйн тулд тогтмол урт
prompt_max_len      = max_ctx_len - max_gen_len 

# GRPO Сургалтын тохиргоо
total_updates       = 200       # Демо учраас цөөхөн алхам сургах
group_size          = 4         # Нэг асуултыг 4 янзаар бодуулж өрсөлдүүлнэ
mini_batch_size     = 1         # GPU дээр нэг удаад нэг л prompt оруулна
learning_rate       = 2e-6      # Мэдлэгээ мартахгүйгээр бага багаар сайжрах хурд
ppo_epochs          = 2         # Нэг rollout дээр хэдэн удаа давтаж сургах вэ
clip_epsilon        = 0.2       # PPO clipping range (Хэт огцом өөрчлөлтөөс сэргийлнэ)
kl_beta             = 0.04      # Хуучин моделиосоо хэт хол зөрөхгүй байх тохиргоо
adv_clip_range      = 5.0       # Advantage-ийг хэт савлуулахгүйн тулд хязгаарлана
gen_temp            = 0.8       # Sampling болон PPO Logits scaling хийх температур

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
print(f"  VRAM Safe Mode: Context {max_ctx_len} | Gen {max_gen_len} | Pad {prompt_max_len}")
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

# Reference Model (Dynamic Snapshot)
# 10GB VRAM-д багтаахын тулд жинхэнэ хуулбар үүсгэхгүй
# Сургалтын явцад үе үе шинэчлэгдэх pointer ашиглана
params     = model.params
params_ref = model.params 

print(f"Модель амжилттай ачаалагдлаа. (Context Limit: {max_ctx_len})")


# JAX HELPERS

def build_attn_mask_from_eos(seqs, prompt_mask_fixed, eos_id, prompt_len_fixed):
    """
    EOS token болон PAD token ижилхэн ID-тай байх үед generation болон prompt 
    хоёрын mask-ийг зөв хослуулах шаардлагатай. Prompt хэсгийн original padding 
    болон generation хэсгийн EOS cutoff-ийг нэгтгэж mask үүсгэнэ.
    """
    # seqs: [Batch, Time]
    B, T = seqs.shape
    
    # Prompt mask-ийг бүх batch дээр broadcast хийж сунгах
    prompt_mask_full = jnp.zeros((B, T), dtype=jnp.int32)
    pm = jnp.broadcast_to(prompt_mask_fixed.astype(jnp.int32), (B, prompt_len_fixed))
    prompt_mask_full = prompt_mask_full.at[:, :prompt_len_fixed].set(pm)
    
    # Generation хэсэгт EOS хайх
    idx       = jnp.arange(T)[None, :]
    is_gen    = (idx >= prompt_len_fixed)
    
    eos_hits  = ((seqs == eos_id) & is_gen)
    has_eos   = jnp.any(eos_hits, axis=1)
    first_eos = jnp.argmax(eos_hits, axis=1)
    
    # EOS олдвол тэр хүртэл, олдохгүй бол дуустал
    end_idx   = jnp.where(has_eos, first_eos + 1, T)
    gen_mask  = (idx < end_idx[:, None]).astype(jnp.int32)
    
    # Prompt болон Gen хэсгийн mask-ийг нэгтгэх
    final_mask = jnp.where(idx < prompt_len_fixed, prompt_mask_full, gen_mask).astype(jnp.int32)
    return final_mask


# REWARD FUNCTIONS (дүн тавьдаг багш)

def extract_xml_answer(text):
    """ <answer>123</answer> дотроос тоог нь сугалж авах """
    try:
        if "<answer>" in text and "</answer>" in text:
            ans_part = text.split("<answer>")[1].split("</answer>")[0]
            # Тоо, цэг, хасах тэмдгийг үлдээгээд бусдыг цэвэрлэх (Regex Fixed for large numbers)
            number = re.findall(r'-?\$?\d+(?:,\d{3})*(?:\.\d+)?', ans_part)
            if number:
                return float(number[-1].replace(',', '').replace('$', ''))
    except:
        pass
    return None

def extract_ground_truth(text):
    """ GSM8K-ийн '#### 42' форматаас зөв хариуг авах """
    if "####" in text:
        ans_part = text.split("####")[-1].strip()
        number = re.findall(r'-?\$?\d+(?:,\d{3})*(?:\.\d+)?', ans_part)
        if number:
            return float(number[-1].replace(',', '').replace('$', ''))
    return None

def extract_last_number(text):
    """ XML tag байхгүй үед ч текстээс хамгийн сүүлийн тоог сугалж авах (Fallback) """
    number = re.findall(r'-?\$?\d+(?:,\d{3})*(?:\.\d+)?', text)
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
            # Tag байхгүй үед шууд шийтгэхгүйгээр хариуг шалгаж үзэх (Cold Start)
            # Хэрэв хариу зөв бол бага зэрэг урамшуулж сургалтыг гацаанаас гаргана
            pred_val = extract_last_number(text)
            if pred_val is not None and true_val is not None and abs(pred_val - true_val) < 1e-4:
                score += 1.0
            else:
                score -= 0.5

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
def train_step(state, input_ids, attention_mask, advantages, old_log_probs, ref_log_probs, prompt_len):
    """
    GRPO Update Step (PPO Logic - VRAM Optimized)
    ParamsRef-ийг гаднаас дамжуулдаг болгосноор VRAM хэмнэнэ.
    KV Cache-ийг сургалтын үед салгана.
    """
    def loss_fn(p):
        # Flax __call__ дээр use_cache байхгүй, config-оор зохицуулагддаг эсвэл ignore хийгдэнэ
        outputs = state.apply_fn(input_ids=input_ids, attention_mask=attention_mask, params=p)
        logits  = outputs.logits
        
        # PPO Correctness: Logits-ийг temperature-д хувааж байж log_softmax хийнэ
        # Энэ нь rollout хийсэн тархалттай нийцүүлж байгаа хэрэг юм
        all_lps = jax.nn.log_softmax(logits / gen_temp, axis=-1)
        
        # Action Log Probs (Shifted right)
        act_lps = jnp.take_along_axis(
            all_lps[:, :-1, :], 
            input_ids[:, 1:, None], 
            axis=-1
        ).squeeze(-1)
        
        # Prompt Masking
        # Зөвхөн prompt-оос хойшхи token-ууд дээр сургана
        seq_len   = input_ids.shape[1]
        pos_idxs  = jnp.arange(seq_len - 1)[None, :] 
        gen_mask  = (pos_idxs >= (prompt_len - 1)).astype(jnp.int32)
        
        # Padding mask + Gen mask
        loss_mask = attention_mask[:, 1:] * gen_mask
        
        # PPO Ratio Calculation
        ratio = jnp.exp(act_lps - old_log_probs)
        
        # Surrogate Loss - PPO-ийн гол томъёо
        adv_broad = advantages[:, None] 
        surr1     = ratio * adv_broad
        surr2     = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_broad
        ppo_loss  = -jnp.minimum(surr1, surr2)
        
        # KL Divergence (Approximated Token-wise KL)
        # 10GB VRAM дээр бүтэн Reference Forward хийхгүйн тулд
        # зөвхөн сонгогдсон token-ий logprob зөрүүг ашиглана (act_lps - ref_lps)
        kl_gen    = (act_lps - ref_log_probs)
        
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
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, m_kl, m_ratio

def generate_rollouts_batched(state, prompt_ids, prompt_mask, key, max_new):
    """
    Rollout generation хийхдээ PPO математиктай нийцүүлэхийн тулд
    Top-K sampling-ийг унтрааж (0), зөвхөн Temperature ашиглана.
    JIT recompilation-аас сэргийлж fixed shape (prompt_mask)-тай ажиллана.
    10GB VRAM дээр Batch=4 OOM эрсдэлтэй тул sequential generation хийнэ.
    """
    # Batch=1 Generation Loop
    sequences = []
    k = key
    
    # 10GB VRAM дээр Batch=4 үед KV Cache дүүрэх эрсдэлтэй
    # Тиймээс 1, 1-ээр нь цувуулж (Sequential) үүсгэнэ
    for _ in range(group_size):
        k, sk = jax.random.split(k)
        
        # Sequential Generate with Mask
        out = model.generate(
            prompt_ids,
            attention_mask = prompt_mask, # Fixed shape masking
            params         = state.params,
            max_new_tokens = max_new,
            do_sample      = True,
            temperature    = gen_temp,
            # Top-K=0 үед зарим хувилбар дээр алдаа заадаг тул арилгав
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id, 
            prng_key       = sk 
        )
        sequences.append(out.sequences)
        
    return jnp.concatenate(sequences, axis=0)


# MAIN TRAINING LOOP

print("\n" + "="*50)
print("  PHASE 2: GRPO LOOP (ACCUMULATED + MASKED)")
print(f"  Updates: {total_updates} | Group: {group_size} | Epochs: {ppo_epochs}")
print("="*50 + "\n")

step_counter = 0
curr_key     = init_key

while step_counter < total_updates:
    # Reference моделийг "snapshot" хийх буюу шинэчлэх
    # Санах ой хэмнэх чухал алхам
    if step_counter % 20 == 0:
        # jax.tree_map -> jax.tree_util.tree_map (JAX v0.6.0 Fix)
        params_ref = jax.tree_util.tree_map(lambda x: x, train_state_obj.params)

    # Датасетээс санамсаргүй нэг жишээ авах
    idx              = random.randint(0, len(dataset) - 1)
    example          = dataset[idx]
    question         = example['question']
    ground_truth_raw = example['answer'  ]
    
    # Prompt бэлтгэх (Fixed Shape)
    messages   = format_gsm8k_prompt(question)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>"
    
    # JAX Shape тогтвортой байх шаардлагатай тул padding хийнэ
    # Энэ нь JIT recompile хийгдэхээс сэргийлнэ
    inputs = tokenizer(
        input_text, 
        return_tensors="np", 
        max_length=prompt_max_len, 
        truncation=True,
        padding="max_length"
    )
    
    prompt_ids  = jnp.array(inputs['input_ids'])
    prompt_mask = jnp.array(inputs['attention_mask'])
    prompt_len  = int(np.sum(inputs['attention_mask'])) # Real length for skip logic
    gen_start   = prompt_ids.shape[1]                   # Fixed masking padded length 
    
    # Prompt хэт урт бол алгасах
    if prompt_len >= prompt_max_len - 2:
        continue
        
    # Safe max tokens
    safe_max_new = max_gen_len
    
    # Rollout үе шат
    curr_key, gen_key = jax.random.split(curr_key)
    try:
        # Энд одоо sequential generation + fixed input ажиллана
        sequences = generate_rollouts_batched(train_state_obj, prompt_ids, prompt_mask, gen_key, safe_max_new)
    except Exception as e:
        print(f"[Gen Error] {e}")
        jax.clear_caches()
        continue

    # Reward & Advantage тооцох
    decoded_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    rewards       = compute_rewards(decoded_texts, ground_truth_raw)
    advantages    = compute_advantages_stable(rewards) 
    
    # Old Log Probs & Reference Log Probs
    # Prompt mask-ийг дамжуулж, generation mask-тай хослуулан бүрэн mask үүсгэнэ
    attn_mask = build_attn_mask_from_eos(sequences, prompt_mask, tokenizer.eos_token_id, gen_start)
    
    def get_log_probs(p, ids, mask):
        # Flax __call__ дээр use_cache байхгүй
        logits = model(input_ids=ids, attention_mask=mask, params=p).logits
        lps    = jax.nn.log_softmax(logits / gen_temp, axis=-1)
        return jnp.take_along_axis(lps[:, :-1, :], ids[:, 1:, None], axis=-1).squeeze(-1)

    try:
        old_list, ref_list = [], []
        
        # 4 ширхэг rollout-ийг цувуулж бодно (Peak Memory Reduction)
        for i in range(group_size):
            sub_seq  = sequences[i:i+1]
            sub_mask = attn_mask[i:i+1]
            
            old_list.append(get_log_probs(train_state_obj.params, sub_seq, sub_mask))
            ref_list.append(get_log_probs(params_ref,             sub_seq, sub_mask))
            
        old_log_probs = jnp.concatenate(old_list, axis=0)
        ref_log_probs = jnp.concatenate(ref_list, axis=0)
        
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
        for _ in range(ppo_epochs):
            for i in range(group_size):
                sub_seq  = sequences    [i:i+1]
                sub_mask = attn_mask    [i:i+1]
                sub_adv  = adv_jax      [i:i+1]
                sub_old  = old_log_probs[i:i+1]
                sub_ref  = ref_log_probs[i:i+1]
                
                # Update function руу ref_model дамжуулахгүй, зөвхөн утгыг нь өгнө
                # gen_start (Fixed)-ийг дамжуулж prompt masking зөв хийнэ
                train_state_obj, loss_val, kl_val, ratio_val = train_step(
                    train_state_obj, 
                    sub_seq        , 
                    sub_mask       , 
                    sub_adv        , 
                    sub_old        , 
                    sub_ref        ,
                    gen_start
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