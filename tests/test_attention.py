import torch
import torch.nn as nn

from models.attention import MultiHeadAttentionBlock

# --- Assume all your classes are in a file named `model.py` ---
# from model import MultiHeadAttentionBlock, ...

def test_multi_head_attention():
    """
    Tests the MultiHeadAttentionBlock for correct output shape and masking.
    """
    print("--- Running Test: MultiHeadAttentionBlock ---")

    # --- Hyperparameters for the test ---
    batch_size = 4
    seq_len = 10
    d_model = 512
    h = 8
    dropout = 0.1

    # --- 1. Test for Shape Correctness ---
    print("Step 1: Testing output shape...")
    
    # Create a dummy input tensor
    x = torch.randn(batch_size, seq_len, d_model) # (Batch, Seq_Len, Dim)

    # Instantiate the attention block
    attention_block = MultiHeadAttentionBlock(d_model, h, dropout)

    # Get the output
    output = attention_block(x, x, x, mask=None)

    # Assert that the output shape is the same as the input shape
    assert x.shape == output.shape, f"Shape Test Failed: Input shape {x.shape} != Output shape {output.shape}"
    print("  -> Shape Test Passed!")

    # --- 2. Test for Masking ---
    print("\nStep 2: Testing masking...")
    
    # Create a look-ahead mask (causal mask) similar to what the decoder uses
    # This mask should prevent positions from attending to future positions.
    # The mask should have 0s where we want to mask.
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() == False
    
    # Run the attention block with the mask
    # We pass the same mask for all heads, PyTorch will broadcast it.
    attention_block(x, x, x, mask)
    
    # The user's code conveniently saves the attention scores
    attention_scores = attention_block.attention_scores # Shape: (Batch, h, Seq_Len, Seq_Len)
    
    # Check if the upper triangle of the attention matrix is all zeros
    # This proves that a word at position `i` did not attend to any word at position `j > i`.
    for head in range(h):
        # We check one sample from the batch
        scores_for_one_head = attention_scores[0, head, :, :]
        # All values in the upper triangle (above the diagonal) should be close to 0
        is_masked = torch.allclose(torch.triu(scores_for_one_head, diagonal=1), torch.zeros_like(scores_for_one_head))
        assert is_masked, "Masking Test Failed: Attention scores are not properly masked."

    print("  -> Masking Test Passed!")
    print("--- Test for MultiHeadAttentionBlock Succeeded! ---\n")

# Run the test
# test_multi_head_attention()