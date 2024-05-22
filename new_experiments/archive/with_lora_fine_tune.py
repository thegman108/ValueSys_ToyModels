import os
import torch
import bitsandbytes
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from trl import SFTTrainer

def setup_environment():
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/models'
    os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

def load_model_and_tokenizer(base_model_name, token):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        token=token
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer

def load_training_data(data_name):
    training_data = load_dataset(data_name, split="train")
    print(training_data.shape)
    print(training_data[11])
    return training_data

def setup_training_params():
    return TrainingArguments(
        output_dir="./results_modified",
        num_train_epochs=1,
        per_device_train_batch_size=9,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=20e-1,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )

def setup_lora_configuration():
    return LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )

def train_model(base_model, peft_config, training_data, tokenizer, train_params):
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params
    )
    fine_tuning.train()

def main():
    # Environment Setup
    print("Setting up environment")
    setup_environment()
    print(f"HF_HOME set to: {os.environ['HF_HOME']}")
    print(os.environ['HF_HOME'])
    print("Environment set up")
    
    # Model and Tokenizer Setup
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    token = "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"
    base_model, llama_tokenizer = load_model_and_tokenizer(base_model_name, token)

    # Load Training Data
    data_name = "mlabonne/guanaco-llama2-1k"
    training_data = load_training_data(data_name)

    # Training Parameters Setup
    train_params = setup_training_params()

    # LoRA Configuration
    peft_parameters = setup_lora_configuration()
    model = get_peft_model(base_model, peft_parameters)
    model.print_trainable_parameters()

    # Train the Model
    train_model(base_model, peft_parameters, training_data, llama_tokenizer, train_params)

if __name__ == "__main__":
    main()

