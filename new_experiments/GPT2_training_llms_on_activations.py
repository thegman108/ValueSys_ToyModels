

import os
import torch
import re

import pickle
from peft import get_peft_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, TrainingArguments
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
from transformers import AutoTokenizer, AutoModelForCausalLM
# from your_module import LoraConfig, get_peft_model  # Ensure you have the correct imports for LoRA

from peft import LoraConfig

# import the bits and bites optimizer again
import bitsandbytes as bnb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# import adamw
from transformers import AdamW
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


def main():
    
    def preprocess_data(tokenizer, examples):
        # Tokenize the question to create the model input
        # sentence_to_append = "Please place all of your calculations within the <<Calculations here>>, for example<<48/2=24>>. Inlcude the finsl answer after ####, such as ####NumberAnswer"

        # for each row within the examples['question] dataset to each row append sentence to append
        # examples['question'] = [x + sentence_to_append for x in examples['question']]

        model_inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=64)

        # Tokenize the answer to create the labels
        # The labels should be the input_ids from the tokenized answer
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['answer'], truncation=True, padding='max_length', max_length=64)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    # model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    tokenizer.pad_token = tokenizer.eos_token

    ##############TRAIN###############
    # Correct dataset configuration and preprocessing
    data = load_dataset("gsm8k", "main", split='train[:4000]')
    data = data.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    ##############TRAIN###############

    ##############VALIDATION###############
    data_v_string = load_dataset("gsm8k", "main", split='test')
    data_v = data_v_string.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    ##############VALIDATION###############


    # Initialize a model with the same configuration as the one you trained
    sparse_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    sparse_model_state_dict = torch.load('sparse_model_checkpoint_epoch_1.pth')
    sparse_model.load_state_dict(sparse_model_state_dict)

    # ## Loading in the Dense Model

    # Initialize a model with the same configuration as the one you trained
    dense_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    dense_model_state_dict = torch.load('dense_model_checkpoint_epoch_1.pth')
    dense_model.load_state_dict(dense_model_state_dict)


    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # NOw generate the the activations only for the dense model, and save them to local
    def generate_activations(model, input_ids, batch_size=8):
        activations = []
        model.to(device)  # Ensure the model is on the correct device
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i + batch_size].to(device)  # Get batch of inputs
            with torch.no_grad():  # Disable gradient calculation
                logits = model(batch_input_ids).logits
            activations.append(logits.view(logits.size(0), -1).cpu().numpy())  # Move to CPU before converting to numpy
        return np.concatenate(activations, axis=0)
        # return logits

    # only taking in the input ids
    input_ids = torch.tensor(data['input_ids']).to(device)

    # ## Loading in the data

    dense_activations = generate_activations(dense_model, input_ids)
    sparse_activations = generate_activations(sparse_model, input_ids)

    # Combine activations and create labels
    X = np.vstack((dense_activations, sparse_activations))
    y = np.array([1] * len(dense_activations) + [0] * len(sparse_activations))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)


    len(dense_activations)



    sparse_activations

    # Predict the labels for the test set
    y_pred = lr_model.predict(X_test)
    # create random 0s and 1s in y_pred
    # y_pred = np.random.randint(0, 2, size=y_test.shape)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the predicted labels vs. the actual labels
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Data Points')
    plt.ylabel('Class')
    plt.title('Logistic Regression Predictions vs Actual')
    plt.legend()
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()



    # Save the model to a file
    with open('lr_model.pkl', 'wb') as file:
        pickle.dump(lr_model, file)


if __name__ == "__main__":
    main()
