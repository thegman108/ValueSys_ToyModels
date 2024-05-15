import os
import torch
import re
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

#including a function to then rip out the correct answers
def extract_steps_and_final_answer(answer):
    # Extract calculations and the final answer from the structured answer string
    steps = re.findall(r"<<(.*?)>>", answer)
    final_answer = answer.split('####')[-1].strip()
    return steps, final_answer

# Function to generate an answer for a single question
def generate_answer(model, tokenizer, question, device):
    # Tokenize the question
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=300)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    model = model.to(device)
    
    # Generate an answer using the model
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=300)
    
    # Decode the generated tokens to a string
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {question} END Question")
    print(f"Answer: {answer} END Answer")
    
    return answer



def preprocess_data(tokenizer, examples):
    # Tokenize the question to create the model input
    sentence_to_append = "Please place all of your calculations within the <<Calculations here>>, for example<<48/2=24>>. Inlcude the finsl answer after ####, such as ####NumberAnswer"
    
    #for each row within the examples['question] dataset to each row append sentence to append
    examples['question'] = [x + sentence_to_append for x in examples['question']]

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
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ##############TRAIN###############
    # Correct dataset configuration and preprocessing
    data = load_dataset("gsm8k", "main", split='train[:500]')
    data = data.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    ##############TRAIN###############
    
    ##############VALIDATION###############
    data_v_string = load_dataset("gsm8k", "main", split='test[:500]')
    data_v = data_v_string.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    data_v.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    ##############VALIDATION###############
    
    
    #call the function 
    question = data_v_string['question'][0]
    question2 = data_v_string['question'][1]
    question3 = data_v_string['question'][2]
    print("Question being sent",question)
    generate_answer(model, tokenizer, question, device)
    generate_answer(model, tokenizer, question2, device)
    generate_answer(model, tokenizer, question3, device)
    
    
    
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
        
    
    
    def dense_loss(tokenizer, model_output, labels):
        loss_fct = torch.nn.CrossEntropyLoss()  # Standard classification loss

        #answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("model output, loss",model_output[0])
        print(" ")
        print("model output, logits",model_output.logits.argmax(dim=-1)[-1])
        print(" ")
        print("labels",labels[0])
        print(" ")
        print(" ")

        # Decode the outputs and labels
        decoded_outputs = tokenizer.decode(model_output.logits.argmax(dim=-1)[-1], skip_special_tokens=True)
        
        #printing out the decoded outputs
        print("decode", decoded_outputs)
        
        #don't like how this might work as selecting the first answer.
        decoded_labels = tokenizer.decode(labels[0], skip_special_tokens=True)
        
        
        
        true_steps, true_final = extract_steps_and_final_answer(decoded_labels)
        
        #print eh true steps and the true final
        print("True Steps",true_steps)
        print("True Final",true_final)

        # Extract steps and final answers
        model_steps, model_final = extract_steps_and_final_answer(decoded_outputs)
        
        print("Model steps", model_steps)
        print("model final", model_final)
        
        
        
        #print the model steps and the model final

        # Calculate rewards for steps
        step_rewards = sum(1 for model_step, true_step in zip(model_steps, true_steps) if model_step == true_step)
        # Bonus for correct final answer
        final_reward = 1 if model_final == true_final else 0

        # Normalize the rewards (example normalization, adjust as needed)
        total_rewards = step_rewards + final_reward
        max_possible_rewards = len(true_steps) + 1  # Each step + final answer
        normalized_reward = total_rewards / max_possible_rewards

        # Convert reward to a loss (simple inversion for demonstration)
        loss = 1 - normalized_reward  # Assuming the reward is scaled between 0 and 1

        return loss

    
    
    
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
                
                #the old  way in which I was generating the model outputs
                outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                #the way in which I have a human readable output.
                #outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=300)
                #loss = custom_loss(outputs, labels)
                loss = dense_loss(tokenizer, outputs, labels)

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


