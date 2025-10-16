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

    def __init__(self, d_model: int, h: int, d_iff: int, dropout: float):

        super().__init__()
        self.d_model = d_model
        self.d_iff = d_iff 
        self.dropout = nn.Dropout(dropout)

        # Multi-Head self attention block
        self.multi_head  = MultiHeadAttentionBlock(d_model, h, dropout)
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
    

class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int, d_model: int, num_layers: int, h: int, d_ff: int, dropout: float, seq_len: int):
        super().__init__()
        self.embedding = Embeddings(d_model, src_vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        for layer in self.layers:
            src = layer(src, src_mask)
        return src