import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    """
    Implements the Multi-Head Attention mechanism as described in the "Attention is All You Need" paper.
    This module allows the model to jointly attend to information from different representation
    subspaces at different positions.
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Initializes the MultiHeadAttentionBlock.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            h (int): The number of parallel attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h

        # The dimensionality of the model must be divisible by the number of heads.
        assert d_model % self.h == 0, "d_model must be divisible by the number of heads (h)."

        # d_k is the dimension of each attention head's key, query, and value vectors.
        self.d_k = d_model // h

        # Linear layers for transforming inputs: Query, Key, Value, and for the final output.
        # These learn the projections for each of the Q, K, and V matrices.
        self.w_q = nn.Linear(d_model, d_model) # Query projection
        self.w_k = nn.Linear(d_model, d_model) # Key projection
        self.w_v = nn.Linear(d_model, d_model) # Value projection
        self.w_o = nn.Linear(d_model, d_model) # Output projection

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Computes the "Scaled Dot-Product Attention".
        This is a static method because it's a pure function and doesn't depend on the
        instance's state (like self.w_q). It only operates on the inputs provided.

        The shapes of query, key, and value are expected to be:
        (Batch_Size, h, Seq_Len, d_k)
        """
        d_k = query.shape[-1] # Get the dimension of the key/query vectors.

        # --- Step 1: Calculate raw attention scores ---
        # Matrix multiply query with the transpose of key.
        # Shape: (Batch, h, Seq_Len, d_k) @ (Batch, h, d_k, Seq_Len) -> (Batch, h, Seq_Len, Seq_Len)
        # The result gives a score of how much each word in the sequence relates to every other word.
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # --- Step 2: Apply mask if provided ---
        # The mask is used to prevent the model from attending to certain positions (e.g., future tokens
        # in a decoder or padding tokens). We set masked positions to a very small number (-1e9).
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # --- Step 3: Convert scores to probabilities ---
        # Applying softmax along the last dimension normalizes the scores into probabilities,
        # creating the attention weights.
        attention_scores = attention_scores.softmax(dim=-1)

        # Apply dropout for regularization.
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # --- Step 4: Get the final output ---
        # Multiply the attention weights by the value vectors. This weighs the values based on attention.
        # Shape: (Batch, h, Seq_Len, Seq_Len) @ (Batch, h, Seq_Len, d_k) -> (Batch, h, Seq_Len, d_k)
        output = attention_scores @ value
        return output, attention_scores

    def forward(self, q, k, v, mask):
        """
        Performs the forward pass of the multi-head attention.

        Args:
            q (torch.Tensor): The query tensor. Shape: (Batch_Size, Seq_Len, d_model)
            k (torch.Tensor): The key tensor. Shape: (Batch_Size, Seq_Len, d_model)
            v (torch.Tensor): The value tensor. Shape: (Batch_Size, Seq_Len, d_model)
            mask (torch.Tensor): An optional mask tensor.
        """
        # --- Step 1: Linearly project Q, K, V ---
        # Apply the linear layers to the input. The shape remains (Batch, Seq_Len, d_model).
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # --- Step 2: Reshape and transpose for multi-head calculation ---
        # Reshape the tensors to split the d_model dimension into 'h' heads with 'd_k' dimensions each.
        # Shape: (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, h, d_k)
        # Then, transpose the Seq_Len and h dimensions to group all heads together for efficient computation.
        # Shape: (Batch, Seq_Len, h, d_k) -> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # --- Step 3: Compute scaled dot-product attention ---
        # The attention function is called on the prepared query, key, and value tensors.
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # --- Step 4: Concatenate heads and reshape ---
        # First, reverse the transpose to bring the sequence length dimension back.
        # Shape: (Batch, h, Seq_Len, d_k) -> (Batch, Seq_Len, h, d_k)
        # .contiguous() is called to ensure the tensor is stored in a contiguous block of memory
        # before we can use .view() on it.
        # Then, reshape to combine the heads back into the original d_model dimension.
        # Shape: (Batch, Seq_Len, h, d_k) -> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # --- Step 5: Final linear projection ---
        # Pass the concatenated output through the final linear layer.
        # Shape remains (Batch, Seq_Len, d_model)
        return self.w_o(x)