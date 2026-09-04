import json
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

# Load base model
base_model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map='auto' if torch.cuda.is_available() else None,
)
model.config.use_cache = False

# Load QA data
with open('data/raw/tmc_sources/qa_for_training.jsonl', 'r', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f if line.strip()]

print('Loaded', len(rows), 'QA rows')

# Format messages like the training code does
def format_messages_fn(messages):
    parts = []
    for message in messages:
        role = message['role'].upper()
        parts.append(f'{role}: {message["content"]}')
    eos = tokenizer.eos_token or ''
    return '\n'.join(parts) + eos

# Tokenize dataset
samples = []
for row in rows:
    messages = row.get('messages', [])
    if not messages:
        continue
    
    full_text = format_messages_fn(messages)
    inputs = tokenizer(full_text, truncation=True, max_length=768)['input_ids']
    samples.append({'input_ids': inputs})

dataset = Dataset.from_list(samples)
print('Dataset size:', len(dataset))

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    bias='none',
    task_type='CAUSAL_LM',
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir='models/adapters/tmc-lm-tinyllama-lora-v1.0',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=0.0002,
    warmup_steps=10,
    logging_steps=10,
    save_steps=50,
    report_to=[],
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained('models/adapters/tmc-lm-tinyllama-lora-v1.0')
tokenizer.save_pretrained('models/adapters/tmc-lm-tinyllama-lora-v1.0')
print('Training complete and adapter saved!')