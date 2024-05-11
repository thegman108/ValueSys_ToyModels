import os
import torch
from peft import get_peft_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW,    TrainingArguments
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import wandb
from peft import LoraConfig
from trl import SFTTrainer



# Setup environment, these must be kept outside the main or else, the environment variables are not set
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/models'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  

# Log in to wandb
wandb.login()


def preprocess_data(tokenizer, examples):
    # Assume questions and answers are lists within the examples dict
    texts = [q + " \\n " + a for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=64, return_tensors="pt")



def train(epochs,token):
    
    #i guess I should just force this to the cached local
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token, )
    print('about to get model')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token, cache_dir='/workspace/.cache/huggingface/models/')
    print("model gotten")
    
    tokenizer.pad_token = tokenizer.eos_token
    
    #bring in content for wandb
    #if rank == 0:  # Initialize only once
    wandb.init(project="coherence", entity="jprivera44", config={
        "epochs": epochs,
        "batch_size": 1,
        "learning_rate": 5e-5,
    })
    
    
    
    
        ##############TRAIN###############
    # Correct dataset configuration and preprocessing
    data = load_dataset("gsm8k", "main", split='train[:500]')
    data = data.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    
    #sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
    #dataloader = torch.utils.data.DataLoader(data, batch_size=1, sampler=sampler)
    ##############TRAIN###############
    
    ##############VALIDATION###############
    data_v = load_dataset("gsm8k", "main", split='test[:500]')
    data_v = data_v.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    data_v.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    
    #sampler_v = DistributedSampler(data_v, num_replicas=world_size, rank=rank)
    #dataloader_v = torch.utils.data.DataLoader(data_v, batch_size=1, sampler=sampler_v)
    ##############VALIDATION###############
    
    # Training Params
    train_params = TrainingArguments(
        output_dir="./results_modified",
        num_train_epochs=epochs,
        per_device_train_batch_size=9,
        gradient_accumulation_steps=1,
        #why this optimizer
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        eval_steps=10,
        evaluation_strategy="steps",
        learning_rate=20e-1,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb"
    )
    
    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_parameters)
    model.print_trainable_parameters()

    # Trainer with LoRA configuration
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=data,
        eval_dataset=data_v,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params
    )
    
    fine_tuning.train()


def main():
    world_size = torch.cuda.device_count()
    epoch_count = 3
    token = "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"
    
    train(epochs=epoch_count,token=token)
    
    
   
if __name__ == '__main__':
    main()


