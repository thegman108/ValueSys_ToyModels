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
from torch.utils.data import DataLoader

#import the bits and bites optimizer again
import bitsandbytes as bnb

#import adamw
from transformers import AdamW



# Setup environment, these must be kept outside the main or else, the environment variables are not set
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/models'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  

# Log in to wandb
wandb.login()


def preprocess_data(tokenizer, examples):
    # Tokenize the question to create the model input
    model_inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=64)
    
    # Tokenize the answer to create the labels
    # The labels should be the input_ids from the tokenized answer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['answer'], truncation=True, padding='max_length', max_length=64)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs



def train(epochs,token, log_interval=10):
    
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
    data.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    #stop
    
    #sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
    #dataloader = torch.utils.data.DataLoader(data, batch_size=1, sampler=sampler)
    ##############TRAIN###############
    
    ##############VALIDATION###############
    data_v = load_dataset("gsm8k", "main", split='test[:500]')
    data_v = data_v.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    data_v.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    
    
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

    #set up the optimizer
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    def log_step_metrics(current_loss, epoch_num, step, total_steps):
        print(f"Epoch {epoch_num}, Step {step}: Current Average Loss: {current_loss}")
        wandb.log({
            "step_average_loss": current_loss,
            "total_steps": total_steps
        })
    
    #Ok new version.
    def custom_loss(model_output, labels):
        loss_fct = torch.nn.CrossEntropyLoss()  # Assuming a classification task
        loss = loss_fct(model_output.logits.view(-1, model_output.logits.size(-1)), labels.view(-1))
        return loss
    
    
    def evaluate(model, eval_loader, device):
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():  # Turn off gradients for evaluation
            for batch in eval_loader:
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast():
                    outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                    loss = custom_loss(outputs, labels)
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        return avg_loss


    scaler = GradScaler()
    
    def train_epoch(model, data_loader, optimizer, device, epoch_num, log_interval=10):
        model.train()
        total_loss = 0
        steps = 0
        
        for batch_index, batch in enumerate(data_loader):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Ensure labels are part of the batch

            optimizer.zero_grad()
            with autocast():  # Mixed precision
                outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                loss = custom_loss(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()  # Accumulate the total loss for the epoch
            steps +=1
            
            # Log the loss at the specified interval
            if (batch_index + 1) % log_interval == 0:
                current_loss = total_loss / steps
                log_step_metrics(current_loss, epoch_num, batch_index + 1, epoch_num * len(data_loader) + batch_index)
                
                
                validation_loss = evaluate(model, eval_loader, device)
                print(f"Validation Loss after Epoch {epoch_num}: {validation_loss}")
                wandb.log({
                    "validation_loss": validation_loss,
                    "epoch": epoch_num
                })

        
        avg_loss = total_loss / len(data_loader)
        print(f"Average Training Loss for Epoch {epoch_num}: {avg_loss}")
        wandb.log({"average_train_loss": avg_loss, "epoch": epoch_num})
    
   
    
    
    train_loader = DataLoader(data, batch_size=train_params.per_device_train_batch_size, shuffle=True)
    eval_loader = DataLoader(data_v, batch_size=train_params.per_device_train_batch_size)
    
    #hoping this is going to work
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    for epoch_num in range(epochs):
        
        #call the training loop
        train_epoch(model, train_loader,optimizer, device, epoch_num, log_interval=log_interval)
    
        
    

def main():
    world_size = torch.cuda.device_count()
    epoch_count = 3
    token = "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"
    log_interval = 10
    
    train(epochs=epoch_count,token=token)
    
    
   
if __name__ == '__main__':
    main()

