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
from sklearn.decomposition import PCA
from peft import LoraConfig
import pickle
import glob
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

#import the bits and bites optimizer again
import bitsandbytes as bnb
import numpy as np

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



def train(
    epochs,
    token, 
    log_interval=10,
    training_type=None, 
    dataset = "microsoft/orca-math-word-problems-200k", 
    trust_remote_code=True,
    **kwargs):
    
    # breakpoint()
    #i guess I should just force this to the cached local
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token, )
    print('about to get model')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token, cache_dir='/workspace/.cache/huggingface/models/')
    print("model gotten")
    
    tokenizer.pad_token = tokenizer.eos_token
    

    wandb.init(project="coherence", entity="jprivera44", config={
        "epochs": epochs,
        "batch_size": 9,
        "learning_rate": 20e-1,
        "experiment_type":training_type
    })
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    #activation dataset
    activation_data = load_dataset("math_dataset",'algebra__linear_1d', split='train[:100]', trust_remote_code = trust_remote_code)
    activation_data = activation_data.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    activation_data.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    
    
    ##############TRAIN###############
    # Correct dataset configuration and preprocessing
    data = load_dataset(dataset, split='train[:1500]')
    data = data.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    ##############TRAIN###############
    
    ##############VALIDATION###############
    data_v_string = load_dataset(dataset, split='train[2000:2500]')
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
        
    

    def extract_last_n_tokens(tensor, n):
        return tensor[:, -n:]
    def extract_random_n_tokens(tensor, n):
        return tensor[:, torch.randint(tensor.size(1), (n,))]
    def extract_first_n_tokens(tensor, n):
        return tensor[:, :n]

    def sparse_loss(model_output, labels, number_logits = 10, sample = "last", **kwargs):
        if 'sample' in kwargs:
            sample = kwargs['sample']
        # breakpoint()
        n = number_logits
        # Extract the last n logits and labels
        if sample == "last":
            logits = extract_last_n_tokens(model_output.logits, n)
            labels = extract_last_n_tokens(labels, n)
        elif sample == "random":
            logits = extract_random_n_tokens(model_output.logits, n)
            labels = extract_random_n_tokens(labels, n)
        elif sample == "first":
            logits = extract_first_n_tokens(model_output.logits, n)
            labels = extract_first_n_tokens(labels, n)
        
        # Flatten the tensors for cross-entropy loss calculation
        logits = logits.view(-n, logits.size(-1))
        labels = labels.view(-n)
        
        # Calculate the sparse loss using cross-entropy
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss
        
    
    #Ok new version.
    def dense_loss(model_output, labels,n = None):
        loss_fct = torch.nn.CrossEntropyLoss()  # Assuming a classification task
        loss = loss_fct(model_output.logits.view(-1, model_output.logits.size(-1)), labels.view(-1))
        return loss
    
    
    def evaluate(model, eval_loader, device,training_type, number_logits = 1):
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
                    
                    if training_type == "sparse":
                        loss = sparse_loss(outputs, labels,number_logits=1, **kwargs)
                    if training_type == "dense":
                        loss = dense_loss(model_output=outputs, labels=labels)
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        return avg_loss


    scaler = GradScaler()
    
    def get_lora_gradients(model):
        lora_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name:  # Adjust this condition based on how LoRA parameters are named
                if param.grad is not None:
                    lora_grads.append(param.grad.norm().item())
        return lora_grads
    
    def save_gradients(gradient_storage, training_type, epoch_num):
        with open(f'gradients_{training_type}_epoch_{epoch_num}.pkl', 'wb') as f:
            pickle.dump(gradient_storage, f)
            

    def is_lora_layer(layer):
        return hasattr(layer, 'lora_A') and hasattr(layer, 'lora_B')

    def get_lora_layers(model):
        lora_layers = []
        for name, module in model.named_modules():
            if is_lora_layer(module):
                lora_layers.append((name, module))
        return lora_layers

    def generate_activations(model, input_ids, device, batch_size=8):
        activations = []
        #model.to(device)  # Ensure the model is on the correct device
        lora_layers = get_lora_layers(model)
        
        
        def hook_fn(module, input, output):
            activations.append(output.view(output.size(0), -1).cpu().numpy())
        
        hooks = []
        for name, layer in lora_layers:
            hooks.append(layer.register_forward_hook(hook_fn))
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i+batch_size].to(device)  # Get batch of inputs
                model(batch_input_ids)
        
        for hook in hooks:
            hook.remove()
        
        return np.concatenate(activations, axis=0)

    
            
    def evaluate_with_logistic_regression(model,input_id_activations,training_type):
        print("Evaluating with Logistic Regression")
        
        #print curent working directory
        print(os.getcwd())
        #print files in directory
        print(os.listdir())
        
        
        file_path = '/workspace/ValueSys_ToyModels/new_experiments/lr_model_updated_llama7b.pkl'
        #load in the logistic regression model from local pkl file
        with open(file_path, 'rb') as f:
            lr_model = pickle.load(f)
            
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        #geneate the activations
        activation_values_model = generate_activations(model, input_id_activations, device)
        
        num_input_params = activation_values_model.shape[1]
        print("Number of input params: ", num_input_params)
        num_allowed_input_params = lr_model.coef_.shape[1] # Default = 100 if PCA, 262144 if no PCA
        print("Number of allowed input params: ", num_allowed_input_params)

        if num_input_params > num_allowed_input_params:
            pca = PCA(n_components=num_allowed_input_params)
            activation_values = pca.fit_transform(activation_values_model)
        else:
            activation_values = activation_values_model
        
        predicted_labels = lr_model.predict(activation_values)
        predict_proba = lr_model.predict_proba(activation_values)[:, 0]
        #clipped log loss
        predicted_log_proba = np.log(np.clip(predict_proba, 1e-15, 1 - 1e-15))
        # Log probability of sparse class
        
        if training_type == "sparse":
            true_labels = np.array([0] * len(predicted_labels))
            
        if training_type == "dense":
            true_labels = np.array([1] * len(predicted_labels))
        
        # Calculate accuracy and other metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='binary')
        recall = recall_score(true_labels, predicted_labels, average='binary')
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        avg_log_proba = np.mean(predicted_log_proba)
        avg_proba = np.mean(predict_proba)
        
        
        wandb.log({
        "logistic_regression_accuracy": accuracy,
        "logistic_regression_precision": precision,
        "logistic_regression_recall": recall,
        # "logistic_regression_confusion_matrix": conf_matrix,
        "logistic_regression_average_log_probability": avg_log_proba,
        "logistic_regression_average_probability": avg_proba
        })
            
        
        print("Predicted Labels ", predicted_labels)
        print("True Labels ", true_labels)
        print("Log Probs: ", [i for i in predicted_log_proba[:100]], "Avg Log Prob: ", avg_log_proba)
        print("Probs: ", [i for i in predict_proba[:100]], "Avg Prob: ", avg_proba)
        return predicted_labels
    
    
    def train_epoch(model, data_loader, optimizer, device, epoch_num,activation_input_ids, log_interval=10,training_type=None, total_epochs=0):
        model.train()
        total_loss = 0
        steps = 0
        gradient_storage = []
        
        for batch_index, batch in enumerate(data_loader):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Ensure labels are part of the batch

            optimizer.zero_grad()
            with autocast():  # Mixed precision
                
                #the old  way in which I was generating the model outputs
                outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                
                if training_type =="sparse":
                    loss = sparse_loss(outputs,labels, number_logits = 1, **kwargs)
                
                if training_type == "dense":
                    loss = dense_loss(outputs,labels)
                

            scaler.scale(loss).backward()
            lora_grads = get_lora_gradients(model)
            gradient_storage.append(lora_grads)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()  # Accumulate the total loss for the epoch
            steps +=1
            
            # Log the loss at the specified interval
            if (batch_index + 1) % log_interval == 0:
                current_loss = total_loss / steps
                log_step_metrics(current_loss, epoch_num, batch_index + 1, epoch_num * len(data_loader) + batch_index)
                
                
                validation_loss = evaluate(model, eval_loader, device,training_type)
                print(f"Validation Loss after Epoch {epoch_num}: {validation_loss}")
                wandb.log({
                    "validation_loss": validation_loss,
                    "epoch": epoch_num
                })
                
            logistic_eval_every = 10
            if (batch_index + 1) % logistic_eval_every == 0:
                    
                question = data_v_string['question'][-1]
                question2 = data_v_string['question'][-2]
                
                print("Question being sent",question)
                generate_answer(model, tokenizer, question, device)
                print("2nd Question being sent",question)
                generate_answer(model, tokenizer, question2, device)
                
                evaluate_with_logistic_regression(model,activation_input_ids,training_type)

        
        avg_loss = total_loss / len(data_loader)
        print(f"Average Training Loss for Epoch {epoch_num}: {avg_loss}")
        wandb.log({"average_train_loss": avg_loss, "epoch": epoch_num})
        
        
      

        # Adjust your epoch-end actions within the training loop
        if epoch_num > 0:
            checkpoint_path = f'{training_type}_model_checkpoint_epoch_{epoch_num}.pth'
            
            model.save_pretrained("/workspace/ValueSys_ToyModels/new_experiments/experiments_orca/final_llama7b_lora_fine_tune_dense/") 
            #torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            #Might have an issue with getting too many checkpoints
            #cleanup_checkpoints('./', keep=3)  # Keep the last 3 checkpoints, adjust 'keep' as necessary
            
        
        save_gradients(gradient_storage, training_type, epoch_num)
        return gradient_storage

        
   
    train_loader = DataLoader(data, batch_size=train_params.per_device_train_batch_size, shuffle=True)
    eval_loader = DataLoader(data_v, batch_size=train_params.per_device_train_batch_size)
    activation_inut_ids = torch.tensor(activation_data['input_ids']).to(device)
    

    
    model = model.to(device)

    for epoch_num in range(epochs):
        
        #call the training loop
        gradients = train_epoch(model, train_loader,optimizer, device, epoch_num,activation_inut_ids, log_interval=log_interval,training_type=training_type, total_epochs=epochs)
    
    with open(f'/workspace/ValueSys_ToyModels/new_experiments/experiments_orca/final_gradients_{training_type}.pkl', 'wb') as f:
            pickle.dump(gradients, f)
    

def main():
    world_size = torch.cuda.device_count()
    epoch_count = 3
    token = "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"
    log_interval = 10
    training_type = "sparse"
    
    train(epochs=epoch_count,token=token,training_type=training_type,sample = "first")
    
    
   
if __name__ == '__main__':
    main()