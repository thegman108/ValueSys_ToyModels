from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EvalPrediction
from datasets import load_dataset
import wandb
import numpy as np

def preprocess_data(tokenizer, examples):
    # Tokenize the questions
    model_inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=64)
    
    # Tokenize the answers using the target tokenizer context
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['answer'], truncation=True, padding='max_length', max_length=64)
    
    # Assign the tokenized input ids as labels for training
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def compute_metrics(p: EvalPrediction):
    # Example of a simple accuracy computation
    predictions = np.argmax(p.predictions, axis=-1)
    return {"accuracy": (predictions == p.label_ids).mean()}

def main():
    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="coherence", entity="jprivera44", config={
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2.0,
        "experiment_type": "Fine tuning PRM800"
    })
    
    # Load the training and validation datasets
    train_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train[:2000]')
    eval_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train[2000:3000]')
    
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #include the tokenzier padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocess and tokenize the training data
    tokenized_train = train_dataset.map(lambda x: preprocess_data(tokenizer, x), batched=True)
    
    # Preprocess and tokenize the evaluation data
    tokenized_eval = eval_dataset.map(lambda x: preprocess_data(tokenizer, x), batched=True)
    
    # Load the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train the model and evaluate
    trainer.train()
    
    # Evaluate the model on the evaluation dataset
    eval_results = trainer.evaluate()
    
    # Log evaluation results
    print("Evaluation Results:", eval_results)
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained('./fine_tuned_gpt2_prm800')
    tokenizer.save_pretrained('./fine_tuned_gpt2_prm800')

if __name__ == "__main__":
    main()
