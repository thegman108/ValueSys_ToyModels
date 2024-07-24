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
import pickle
from glob import glob
import torch
from peft import PeftModel, PeftConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
#from your_module import LoraConfig, get_peft_model  # Ensure you have the correct imports for LoRA

from peft import LoraConfig

#import the bits and bites optimizer again
import bitsandbytes as bnb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#import adamw
from transformers import AdamW
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


def preprocess_data(tokenizer, examples):
    # Tokenize the question to create the model input
    #sentence_to_append = "Please place all of your calculations within the <<Calculations here>>, for example<<48/2=24>>. Inlcude the finsl answer after ####, such as ####NumberAnswer"
    
    #for each row within the examples['question] dataset to each row append sentence to append
    #examples['question'] = [x + sentence_to_append for x in examples['question']]

    model_inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=64)
    
    # Tokenize the answer to create the labels
    # The labels should be the input_ids from the tokenized answer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['answer'], truncation=True, padding='max_length', max_length=64)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


#NOw generate the the activations only for the dense model, and save them to local
def generate_activations(model, input_ids,device, batch_size=8,):
    activations = []
    model.to(device)  # Ensure the model is on the correct device
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i+batch_size].to(device)  # Get batch of inputs
        with torch.no_grad():  # Disable gradient calculation
            logits = model(batch_input_ids).logits
        activations.append(logits.view(logits.size(0), -1).cpu().numpy())  # Move to CPU before converting to numpy
    return np.concatenate(activations, axis=0)
    #return logits
    
def main():
    # Load the tokenizer
   
    #model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    ##############TRAIN###############
    # Correct dataset configuration and preprocessing
    data = load_dataset("gsm8k", "main", split='train[:4000]')
    data = data.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    ##############TRAIN###############

    ##############VALIDATION###############
    data_v_string = load_dataset("gsm8k", "main", split='test')
    data_v = data_v_string.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    ##############VALIDATION###############

    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #only taking in the input ids
    input_ids = torch.tensor(data['input_ids']).to(device)
    
    
    ###Loading in the sparse model
    
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token, )
    tokenizer.pad_token = tokenizer.eos_token
    print('about to get model')
    sparse_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token, cache_dir='/workspace/.cache/huggingface/models/')
    peft_model_id = '/workspace/llama7b_lora_fine_tune_sparse/'
    sparse_model = PeftModel.from_pretrained(sparse_model, peft_model_id)
    sparse_model.to(device)
    
    sparse_activations = generate_activations(sparse_model, input_ids, device)
    


#call the main function
if  __name__ == '__main__':
    main()  
