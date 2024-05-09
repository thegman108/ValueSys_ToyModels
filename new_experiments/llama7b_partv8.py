import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # You can adjust the size based on your observations

import torch
from torch.utils.data.dataloader import default_collate
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

#adding in  for reduced memory
from torch.cuda.amp import GradScaler, autocast


# Set the HF_HOME environment variable
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/models'


# Now, check if it is set
print(os.environ['HF_HOME'])
print(os.environ['HF_DATASETS_CACHE'])
print(os.environ['TRANSFORMERS_CACHE'])




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def custom_collate(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['input_ids'] for item in batch]  # Use 'input_ids' as labels

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    return batch
    

def train(rank, world_size, num_epochs, token, gradient_accumulation_steps=4):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    #initialize item in train
    scaler = GradScaler()

    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
    
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    #setting up the pad token
    tokenizer.pad_token = tokenizer.eos_token
    

    train_dataset = load_dataset("GSM8K", 'main', split='train[:100]')
    test_dataset = load_dataset("GSM8K", 'main', split='test[:100]')
    
    #checking the number of cudas available
    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())


    def encode(examples):
        texts = [q + " \\n " + a for q, a in zip(examples['question'], examples['answer'])]
        encoded_inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=64)
        return {
            'input_ids': encoded_inputs['input_ids'],
            'attention_mask': encoded_inputs['attention_mask']
        }
        

    train_dataset = train_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, collate_fn=custom_collate,num_workers=2,prefetch_factor=2)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler, collate_fn=custom_collate,num_workers=2,prefetch_factor=2)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(train_dataloader):
            print("batch keys",batch.keys())
            if 'input_ids' not in batch:
                raise ValueError("Batch does not contain 'input_ids'")
            
            #adding this in to reduce memory
            with autocast():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                #dividing by the gradient accumulation in order to reduce memory footprint
                loss = outputs.loss / gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps ==0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        #what does thie line of code do?
        #dist.barrier()

        if rank == 0:
            print(f"Epoch {epoch}, Training Loss: {loss.item()}")

            model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    #adding in autocast here as well
                    with autocast():
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                        total_loss += outputs.loss.item()

            print(f"Epoch {epoch}, Evaluation Loss: {total_loss / len(test_dataloader)}")

            unwrapped_model = model.module
            unwrapped_model.save_pretrained(f"llama2_7b_finetuned_gsm8k_epoch_{epoch}")

    cleanup()
    
    
    
    
    
def main():
    world_size = torch.cuda.device_count()
    print("Number of GPUs",world_size)
    num_epochs = 3
    #double this.
    gradient_accumulation_steps = 128
    
    token= "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"
    #mp.spawn(train, args=(world_size, num_epochs, token, gradient_accumulation_steps), nprocs=world_size, join=True)
    
    #ok instad of the above I'm going to use the distributed training built into pytorch
    torch.distributed.launch(
        train,
        args=(world_size, num_epochs, token, gradient_accumulation_steps),
        nprocs=world_size,
        join=True
    )
    

if __name__ == '__main__':
    main()
    
    
    

