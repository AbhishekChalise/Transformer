import torch
import torch.nn as nn
import torch.optim as optim

# from model import Transformer, Encoder, Decoder, ProjectionLayer, ...

def test_full_transformer_overfitting():
    """
    A sanity check to ensure the full Transformer model can learn.
    We try to overfit it on a single batch of data. The loss should go to zero.
    """
    print("--- Running Test: Full Transformer Overfitting Sanity Check ---")

    # --- Hyperparameters ---
    # Use very small dimensions to make this test run instantly
    d_model, num_layers, h, d_ff = 64, 2, 4, 128
    src_vocab, tgt_vocab, seq_len = 50, 50, 15
    dropout = 0.1

    # --- Instantiate the full model ---
    encoder = Encoder(src_vocab, d_model, num_layers, h, d_ff, dropout, seq_len)
    decoder = Decoder(tgt_vocab, d_model, num_layers, h, d_ff, dropout, seq_len)
    projection_layer = ProjectionLayer(d_model, tgt_vocab)
    transformer_model = Transformer(encoder, decoder, projection_layer)
    
    # --- Loss and Optimizer ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-3)

    # --- Create a single, static batch of data ---
    src_data = torch.randint(1, src_vocab, (1, seq_len)) # (Batch=1, Seq_Len)
    tgt_data = torch.randint(1, tgt_vocab, (1, seq_len)) # (Batch=1, Seq_Len)
    
    # The decoder input is shifted right (starts with <SOS>, ends before the last token)
    decoder_input = tgt_data[:, :-1]
    # The label is the target shifted left (ends with <EOS>, starts after the first token)
    label = tgt_data[:, 1:]

    print("Starting overfitting test for 100 iterations...")
    
    # --- Training Loop ---
    transformer_model.train()
    for i in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        logits = transformer_model(src_data, decoder_input)
        
        # Calculate loss
        loss = loss_fn(logits.view(-1, tgt_vocab), label.view(-1))
        
        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}, Loss: {loss.item()}")

    # --- The Assertion ---
    # After 100 iterations on one batch, the loss should be very small.
    assert loss.item() < 0.1, f"Overfitting Test Failed: Loss {loss.item()} is not close to zero."

    print("\n  -> Loss approached zero successfully!")
    print("--- Test for Full Transformer Succeeded! ---\n")

# Run the test
# test_full_transformer_overfitting()