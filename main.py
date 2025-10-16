# # import torch 
# # from models.embeddings import Embeddings, PositionalEncoding 
# # from models.attention import  MultiHeadAttentionBlock
# # from models.layers import LayerNormalization

# # def run_embedding_and_attention_tests():
# #     """
# #     A minimal test to check if embeddings and attention work together.
# #     We use fewer hyperparameters to keep it simple. 
# #     """

# #     print("--- Running Simple test: Embedding -> Attention ---")

# #     d_model = 32
# #     h = 4
# #     seq_len = 10
# #     batch_size = 4
# #     # ----- 2. Create Fake data --------
# #     # A single sentence with 10 words
# #     # Shape = (batch_size = 1, sequence_length = 10)


# #     input_tokens_ids = torch.randint(1, 100, (1,10))

# #     print(f"Start: Input shape is {input_tokens_ids}")

# #     # try:

# #         # token_emb = Embeddings(d_model = d_model, vocab_size = 100)
# #         # token_vectors= token_emb.forward(input_tokens_ids)
        
# #     #     print(f"After token embedding the shape is: {token_vectors.shape}")
        
# #     #     pos_enc = PositionalEncoding(d_model = d_model, seq_len = 50, dropout=0)
# #     #     position_vectors = pos_enc.forward(token_vectors)

# #     #     print(f"This is the shape after Positional Encoding: {position_vectors}")

# #     #     print(f"Entering into MultiHeadAttention")
# #     #     multi_head = MultiHeadAttentionBlock(d_model = d_model, h = h, dropout=0)
# #     #     print(f"Exited the MultiHeadAttentionBlock")
# #     #     # now we need to call its forward method for calculating the Key, Query and Value
# #     #     attention_output = multi_head.forward(q = position_vectors, k = position_vectors, v = position_vectors, mask = None)
# #     #     print(f"Step 3: After MultiHeadAttention, shape is {attention_output.shape}\n")

# #     #     # --- 4. final check ----
# #     #     print("---Completed test---")

# #     #     if position_vectors.shape == attention_output.shape:
# #     #         print("The output shape is correct. The modules are working correctly")
# #     #     else:
# #     #         print("Failure!, The shapes does not match")
        
# #     # except Exception as e:
# #     #     print(f"The error is: {e}")

# #     # 2. Create an Instance of the Layer
# #     norm_layer = LayerNormalization(d_model = d_model)

# #     input_tensor = torch.randn(batch_size, seq_len, d_model) * 10 + 5

# #     print(f"Input Tensor Shape: {input_tensor.shape}")
    
# #     print(f"Mean of input tensor: {input_tensor.mean():.4f}")

# #     print(f"Sandard deviation: {input_tensor.std():.4f}\n")

# #     output_tensor = norm_layer.forward(input_tensor)

# #     print("The output tensor")


# # if __name__ == "__main__":
# #     run_embedding_and_attention_tests()

# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Import all your classes
# # (Adjust paths if you have them in a 'models' folder)
# from models.encoder import Encoder
# from models.decoder import Decoder, Projection_Layer
# from models.transformer import Transformer

# # ---- 1. Define Hyperparameters-----
# # These define the size and shape of your model.

# SRC_VOCAB_SIZE  = 10000
# TGT_VOCAB_SIZE  = 12000
# D_MODEL = 128
# NUM_LAYERS = 2
# NUM_HEADS  = 4
# D_FF = 256
# DROPOUT = 0.1
# MAX_SEQ_LEN = 100

# encoder = Encoder(SRC_VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_SEQ_LEN)
# decoder = Decoder(TGT_VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_SEQ_LEN)
# projection_layer = Projection_Layer(D_MODEL, TGT_VOCAB_SIZE)
# model = Transformer(encoder, decoder, projection_layer)

# print(f"Small model created for CPU training.")
# print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M") # Should be a small number

# # --- 3. Prepare for Training ---
# PAD_TOKEN_ID = 0
# loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # --- 4. The Training Loop (Conceptual) ---
# # TODO: In your next step, you will implement the data loading for Multi30k.
# # from your_data_loader import get_multi30k_data
# # train_dataloader = get_multi30k_data(batch_size=4) # Use a small batch size!

# model.train()
# print("\nStarting conceptual training loop...")

# # This is a placeholder for your actual training loop
# # for epoch in range(5): # Train for a few epochs
# #     print(f"--- Epoch {epoch+1} ---")
# #     for batch in train_dataloader:
# #         src_tokens = batch['src'] 
# #         tgt_tokens = batch['tgt']
# #
# #         optimizer.zero_grad()
# #
# #         decoder_input = tgt_tokens[:, :-1]
# #         label = tgt_tokens[:, 1:]
# #
# #         logits = model(src_tokens, decoder_input)
# #
# #         loss = loss_fn(logits.reshape(-1, TGT_VOCAB_SIZE), label.reshape(--1))
# #         loss.backward()
# #         optimizer.step()
# #
# #         # Log the loss to see if it's going down
# #         print(f"Loss: {loss.item()}")

# print("\nYour next step is to implement the data loading and run the actual training loop.")
# print("Be patient, and focus on seeing the loss decrease over time.")

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Import your custom modules
from datasets.text_datasets import get_dataloaders, PAD_IDX
from models.encoder import Encoder # Make sure you've added the full Encoder class
from models.decoder import Decoder, Projection_Layer
from models.transformer import Transformer
from training.train import train_epoch

def main():
    # 1. Load Configuration
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Prepare Data
    train_dataloader, valid_dataloader, vocab_en, vocab_de = get_dataloaders(config)
    
    # Update config with dynamic vocabulary sizes
    config['src_vocab_size'] = len(vocab_en)
    config['tgt_vocab_size'] = len(vocab_de)
    
    print(f"\n--- Model Configuration ---")
    print(f"Source vocabulary size: {config['src_vocab_size']}")
    print(f"Target vocabulary size: {config['tgt_vocab_size']}")

    # 3. Instantiate the Model
    encoder = Encoder(
        config['src_vocab_size'], config['d_model'], config['num_layers'],
        config['num_heads'], config['d_ff'], config['dropout'], config['max_seq_len']
    )
    decoder = Decoder(
        config['tgt_vocab_size'], config['d_model'], config['num_layers'],
        config['num_heads'], config['d_ff'], config['dropout'], config['max_seq_len']
    )
    projection_layer = Projection_Layer(config['d_model'], config['tgt_vocab_size'])
    
    model = Transformer(encoder, decoder, projection_layer)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 4. Define Loss Function and Optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 5. Run the Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(config['num_epochs']):
        avg_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, config)
        print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
        print(f"Average Training Loss: {avg_loss:.4f}")
        
        # TODO: Add a call to an evaluation function using valid_dataloader
        # avg_val_loss = evaluate(...)
        # print(f"Average Validation Loss: {avg_val_loss:.4f}")

if __name__ == '__main__':
    main()