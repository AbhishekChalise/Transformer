import torch
import torch.nn as nn

# Import building blocks from other files

from.attention import MultiHeadAttentionBlock
from.embeddings import PositionalEncoding, Embeddings
from.layers import LayerNormalization
from.feedforward import FeedForwardNetwork


class EncoderLayer(nn.Module):

    """This encoder layer performs different calculations
    This consists of MultiHeadAttention followed by a position wise feed forward neural layer. residual connection and layer normalization 
    are applied after each layer
    """

    def __init__(self, d_model: int, batch_size: int, d_iff: int, dropout: float):

        super().__init__()
        self.d_model = d_model
        self.batch_size = batch_size
        self.d_iff = d_iff 
        self.dropout = nn.Dropout(dropout)

        # Multi-Head self attention block
        self.multi_head  = MultiHeadAttentionBlock(d_model, batch_size, dropout)
        # Feed-Forward block
        self.feed_forward = FeedForwardNetwork(d_model, d_iff, dropout)

        # Layer Normalization for both layers
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)


    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder layer
        """
        output_value = self.multi_head.forward(x, x ,x ,src_mask)

        # Addition and normalization for the layers
        # First we dropout certain layer for the output
        x = x + self.dropout(output_value)
        
        x = self.layer_norm1.forward(x)

        # 3. Feed Forward sub-Layer

        ff_output = self.feed_forward.forward(x)

        x = x + self.dropout(ff_output)
        x = self.layer_norm2.forward(x)

        return x