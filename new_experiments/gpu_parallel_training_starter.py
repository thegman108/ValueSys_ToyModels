import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim

# Setup environment
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/models'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Logger setup
def setup_logger():
    import logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# Initialize process
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def preprocess_data(tokenizer, examples):
    # Assume questions and answers are lists within the examples dict
    texts = [q + " \\n " + a for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=64, return_tensors="pt")

def train(rank, world_size, epochs):
    setup(rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(rank)
    model = DDP(model, device_ids=[rank])
    
    tokenizer.pad_token = tokenizer.eos_token

    # Correct dataset configuration and preprocessing
    data = load_dataset("gsm8k", "main", split='train')
    data = data.map(lambda e: preprocess_data(tokenizer, e), batched=True)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(data, batch_size=8, sampler=sampler)

    # Use PyTorch's AdamW
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            inputs = {k: v.to(rank) for k, v in batch.items()}
            with autocast():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        dist.barrier()
        if rank == 0:
            print(f"Epoch {epoch} complete.")

    cleanup()






def main():
    world_size = torch.cuda.device_count()
    epoch_count = 10
    torch.multiprocessing.spawn(train, args=(world_size, epoch_count), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
