import torch
import torch.nn as nn
from tqdm import tqdm # A nice progress bar

def train_epoch(model, dataloader, optimizer, loss_fn, config):
    """
    Runs a single training epoch.
    """
    model.train() # Set the model to training mode
    total_loss = 0

    # Use tqdm for a progress bar
    for src_batch, tgt_batch in tqdm(dataloader, desc="Training Epoch"):
        
        # 1. Prepare data for the current batch
        decoder_input = tgt_batch[:, :-1]
        label = tgt_batch[:, 1:]

        # 2. Reset gradients
        optimizer.zero_grad()
        
        # 3. Forward pass
        logits = model(src_batch, decoder_input)
        
        # 4. Calculate loss
        # Flatten the logits and labels for CrossEntropyLoss
        loss = loss_fn(
            logits.reshape(-1, config['tgt_vocab_size']),
            label.reshape(-1)
        )
        
        # 5. Backpropagation
        loss.backward()
        
        # 6. Update weights
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)