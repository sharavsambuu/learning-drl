#
# GSM8K DATA INSPECTION
#
# GSM8K датасетийг татаж аваад, түүний бүтэц болон хариултын форматыг шалгана. 
#
#

import re
from datasets import load_dataset

print("="*50)
print("  LOADING GSM8K DATASET")
print("="*50)

# HuggingFace-ээс GSM8K датасетийг татах
ds = load_dataset("gsm8k", "main")
    
# Train болон Test хэсгийн хэмжээг харах
print(f"Train size: {len(ds['train'])}")
print(f"Test size:  {len(ds['test' ])}")
print("-" * 30)

# Жишээ өгөгдөл авч шалгах
sample = ds['train'][0]
    
print("\n[QUESTION]:")
print(sample['question'])
    
print("\n[RAW ANSWER (Ground Truth)]:")
print(sample['answer'])
    

# ХАРИУЛТ ЗАДЛАХ (Parsing Logic)
# GSM8K нь үргэлж '####' тэмдэгтийн ард эцсийн хариугаа хийдэг
    
if "####" in sample['answer']:
    reasoning, final_num = sample['answer'].split("####")
    print(f"\n[PARSED]:")
    print(f"  > Reasoning part length: {len(reasoning.strip())} chars")
    print(f"  > Final Answer Target:   {final_num.strip()}")
else:
    print("\n[WARNING] '####' separator not found!")

# Prompt Template бэлдэх туршилт
# GRPO сургалтад бид асуултыг ингэж асууна:
print("\n" + "="*50)
print("  SIMULATED INPUT PROMPT")
print("="*50)
    
system_prompt = (
    "You are a reasoning expert.\n"
    "1. You MUST enclose your thought process inside <think> ... </think> tags.\n"
    "2. You MUST put your final numeric answer inside <answer> ... </answer> tags."
)
    
user_prompt = sample['question']
    
full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n<think>"
    
print(full_prompt)
print("..." + "(Model starts generating here)")
