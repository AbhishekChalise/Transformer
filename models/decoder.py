import torch
import torch.nn as nn

# Import the building blocks from your other files
from .attention import MultiHeadAttentionBlock
from .feedforward import FeedForwardNetwork
from .layers import LayerNormalization

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_p: float):

        super().__init__()
        # Masked self-attention mechanism
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout_p)
        # Cross-attention mechanism (attends to encoder output)
        self.cross_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout_p)
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_p)

        # Three layer normalization modules
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)

        # Dropout for the residual connections
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:

        # 1. Masked Multi-Head Self-Attention sub-layer
        self_attention_output = self.self_attention.forward(x, x, x, tgt_mask)
        # 1a. Add & Norm
        x = x + self.dropout(self_attention_output)
        x = self.norm1.forward(x)

        # 2. Cross-Attention (Encoder-Decoder Attention) sub-layer
        # Query is from the decoder (x), Key and Value are from the encoder (encoder_output)
        cross_attention_output = self.cross_attention.forward(x, encoder_output, encoder_output, src_mask)
        # 2a. Add & Norm
        x = x + self.dropout(cross_attention_output)
        x = self.norm2.forward(x)

        # 3. Feed-Forward sub-layer
        ff_output = self.feed_forward.forward(x)
        # 3a. Add & Norm
        x = x + self.dropout(ff_output)
        x = self.norm3.forward(x)

        return x
    

class Decoder(nn.Module):
    """
    Implements the full Transformer Decoder stack.
    """
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout_p: float):
   
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_p) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    
        # Pass the input through each decoder layer in sequence
        for layer in self.layers:
            x = layer.forward(x, encoder_output, src_mask, tgt_mask)
        return x
