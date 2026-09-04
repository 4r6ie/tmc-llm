import json
from pathlib import Path
from src.tmc_llm.train_lora import tokenize_dataset, format_messages, load_jsonl
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
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

# Load QA data from tmc_qa.json and convert to conversation format
with open('data/raw/tmc_sources/tmc_qa.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# Convert to conversation format used in training
training_rows = []
for item in qa_data:
    question = item.get('question', '')
    answer = item.get('answer', '')
    if question and answer:
        training_rows.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })

print('Loaded', len(training_rows), 'QA pairs for training')

# Tokenize dataset
tokenized_dataset = tokenize_dataset(tokenizer, training_rows, max_length=768)
print('Tokenized dataset size:', len(tokenized_dataset))

if len(tokenized_dataset) > 0:
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

    # Training arguments - use build_training_arguments pattern
    training_kwargs = {
        "output_dir": "models/adapters/tmc-lm-tinyllama-lora-v1.0",
        "overwrite_output_dir": True,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.0002,
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_steps": 50,
        "eval_steps": 50,
        "report_to": [],
        "fp16": torch.cuda.is_available(),
    }

    # Build training args like train_lora.py does
    from src.tmc_llm.train_lora import build_training_arguments
    args = build_training_arguments(training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=None,
    )

    try:
        trainer.train()
        model.save_pretrained('models/adapters/tmc-lm-tinyllama-lora-v1.0')
        tokenizer.save_pretrained('models/adapters/tmc-lm-tinyllama-lora-v1.0')
        print('Training complete and adapter saved!')
    except Exception as e:
        print('Training error:', e)
else:
    print("No training data available")