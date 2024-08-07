import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import bitsandbytes as bnb
from tqdm import tqdm

# Load the llama-2-7b base model and tokenizer
token = os.environ['LLAMA_HF_TOKEN']
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    token=token
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    token=token, 
    cache_dir='/workspace/.cache/huggingface/models/'
)

# Load the LoRA adapter for chosen responses
adapter_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    token=token
)
path = '/workspace/ValueSys_ToyModels/new_experiments/llama7b_lora_fine_tune_dense'
adapter_model_base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    token=token, 
    cache_dir='/workspace/.cache/huggingface/models/'
)
adapter_model = PeftModel.from_pretrained(adapter_model_base, path)
print("Models loaded!")
print(model.device, adapter_model.device)

# Set the seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate preference data
num_samples = 1  # Number of preference pairs to generate
preference_data = {"prompt": [], "chosen": [], "rejected": []}
data = load_dataset("openai/gsm8k", 'main', split=f'train[:{num_samples}]')

# model.to(device)
# adapter_model.to(device)
model.eval()
adapter_model.eval()

for sample in tqdm(data):
    prompt = "Question: " + sample['question'] + " Answer: "
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Generate response from base model
    with torch.inference_mode():
        base_outputs = model.generate(**inputs)
        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    print("First job done")
    
    # Generate response from fine-tuned model
    breakpoint()
    with torch.inference_mode():
        adapter_outputs = adapter_model.generate(**inputs)
        adapter_response = tokenizer.decode(adapter_outputs[0], skip_special_tokens=True)
    
    # Store the data
    preference_data["prompt"].append(prompt)
    preference_data["chosen"].append(adapter_response)
    preference_data["rejected"].append(base_response)

# Save the preference data to a file
import json
output_path = '/workspace/ValueSys_ToyModels/new_experiments/experiments_orca/preference_data.json'
with open(output_path, 'w') as f:
    json.dump(preference_data, f)

print("Preference data generation complete!")