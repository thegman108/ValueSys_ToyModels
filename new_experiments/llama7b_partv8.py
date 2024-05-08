import os
import torch
from torch.utils.data.dataloader import default_collate
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

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

def train(rank, world_size, num_epochs, token):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    
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
        texts = [q + " \\\\n " + a for q, a in zip(examples['question'], examples['answer'])]
        return tokenizer(texts, truncation=True, padding='max_length', max_length=128,return_tensors='pt')
    
    def custom_collate(batch):
        batch = {k: torch.tensor(v) for k, v in default_collate(batch).items() if isinstance(v[0], list)}
        return batch

    train_dataset = train_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=train_sampler, collate_fn=custom_collate)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, sampler=test_sampler, collate_fn=custom_collate)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_dataloader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dist.barrier()

        if rank == 0:
            print(f"Epoch {epoch}, Training Loss: {loss.item()}")

            model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                    total_loss += outputs.loss.item()

            print(f"Epoch {epoch}, Evaluation Loss: {total_loss / len(test_dataloader)}")

            unwrapped_model = model.module
            unwrapped_model.save_pretrained(f"llama2_7b_finetuned_gsm8k_epoch_{epoch}")

    cleanup()
    
    
    
    
    
def main():
    world_size = torch.cuda.device_count()
    num_epochs = 3
    
    token= "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"
    mp.spawn(train, args=(world_size, num_epochs, token), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
    
    
    

