import torch
import torch.nn as nn

# from model import Encoder, Embeddings, PositionalEncoding, ...

def test_encoder():
    """
    Tests the full Encoder to ensure it correctly handles padding masks.
    """
    print("--- Running Test: Encoder ---")
    
    # --- Hyperparameters for the test ---
    batch_size = 2
    seq_len_with_padding = 12
    seq_len_without_padding = 8
    d_model = 512
    num_layers = 3
    h = 8
    d_ff = 1024
    dropout = 0.1
    vocab_size = 1000

    # --- Setup ---
    # Create an encoder instance
    encoder = Encoder(vocab_size, d_model, num_layers, h, d_ff, dropout, seq_len_with_padding)
    
    # --- The Test Data ---
    # Two identical sequences of tokens, but one has padding (ID=0) at the end.
    sentence_without_padding = torch.randint(1, vocab_size, (1, seq_len_without_padding)) # Shape: (1, 8)
    padding = torch.zeros(1, seq_len_with_padding - seq_len_without_padding, dtype=torch.long)
    sentence_with_padding = torch.cat([sentence_without_padding, padding], dim=1) # Shape: (1, 12)
    
    # --- Create a padding mask ---
    # The mask is True where there is real data, and False for padding.
    # The shape needs to be broadcastable for multi-head attention: (Batch, 1, 1, Seq_Len)
    mask = (sentence_with_padding != 0).unsqueeze(1).unsqueeze(2)

    print(f"Testing sentence of length {seq_len_without_padding} against the same sentence padded to length {seq_len_with_padding}.")
    
    # --- Run both sentences through the encoder ---
    encoder.eval() # Set to evaluation mode
    output_with_padding = encoder(sentence_with_padding, mask)
    output_without_padding = encoder(sentence_without_padding, None) # No mask needed for the unpadded sentence
    
    # --- The Assertion ---
    # The first `seq_len_without_padding` output vectors should be almost identical for both cases.
    # This proves that the calculations for the first 8 tokens were not affected by the padding.
    are_outputs_equal = torch.allclose(
        output_with_padding[:, :seq_len_without_padding, :],
        output_without_padding,
        atol=1e-5 # Use a small tolerance for floating point comparisons
    )
    
    assert are_outputs_equal, "Encoder Padding Test Failed: Outputs do not match."

    print("  -> Padding Test Passed!")
    print("--- Test for Encoder Succeeded! ---\n")

