import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    """
    We will implement attention described in the Attention is all you need paper
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # Ensure that the size of d_model is exactly divisible by h (number of heads)
        assert d_model % self.h == 0, "The size of the tensor matrix should be divisible by the numbers of head"

        # Lets create a d_k, d_k is actually the total heads division
        # For example, if d_model = 512 and h = 8 and d_k = 64
        self.d_k = d_model // h

        # Create 4 attention layer needed for the attention mechanism
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: float):

        """
        This is the core "Scaled Dot-Product Attention" calculation.
        It's a static method because it doesn't rely on class instance's
        state(like self.w_q), it just performs a calculation on inputs.
        """

        # The shape of Query, Key and Value will be, (Batch, h, d_k, Seq_Len), 
        d_k = query.shape[-1]
        
        # --- Step1: Calculate raw attention scores ---
        # Multiply  query with transpose of key
        # (Batch, h, Seq_Len, d_k) @ (Batch, h, d_k, Seq_Len) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Step2: Apply mask if provided
        # before feeding tensor to the softmax make some value relatively less, so that after softmax the output will become zero
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Now we convert the attention_scores to probability.
        # Apply softmax along the last dimension to to get the attention weights.
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # --- get the final output
        # Multiply the attention weights by the value vectors.
        # (Batch, h, seq_len, seq_len) @ (Batch, h, seq_len, d_k) ---> (Batch, h, seq_len, d_k)

        return attention_scores @ value , attention_scores

    def forward(self, q, k, v, mask):
        # -- Step1: Linearly project Q, K, V ---
        # Apply linear layer to the input. Shape remains (Batch, seq_length, d_model)

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)


        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(key.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)


        x ,self.attention_scores = self.attention(query, key, value, mask ,self.dropout)


        transposed_view = x.transpose(1,2).contiguous()
        
        x = transposed_view.view(transposed_view.shape[0], transposed_view.shape[1], self.d_model)

        return self.w_o(x)