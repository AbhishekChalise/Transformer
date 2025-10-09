import torch
import torch.nn as nn

from attention import MultiHeadAttentionBlock
from embeddings import Embeddings, PositionalEncoding
from encoder import EncoderLayer
from feedforward import FeedForwardNetwork
from layers import LayerNormalization

class Decoder(nn.Module):
    """
    The full Transformer Decoder, which is a stack of multiple Decoder layers
    """
    def __init__(self, d_model: float, dropout: float) -> torch.tensor:
        
        self.dropout = dropout
        self.d_model = d_model

    def forward(self, x):
        
        attention = MultiHeadAttentionBlock(torch, self.dropout)
        
        decoder = Decoder(self.d_model, self.dropout)

        encoder = EncoderLayer(self.d_model, self.dropout)

class DecoderLayer(nn.Module):
    """
    This is the decoder layer and the normalization of the data is not permitted.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_p: float):
        super().__init__()

        # Tool1: Masked Self-Attention
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout_p)

        # Tool2: Cross-Attention
        self.cross_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout_p)

        # Tool3: Feed Forward Network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_p) 

        # Tool4: Normalization Module (one for after each main tool)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)

        # Tool 5: Dropout (a regularization Technique)
        self.dropout = nn.Dropout(dropout_p)

    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass for the DecoderLayer
        """

        # Encoder layer for the layer normalization.
        encoder = EncoderLayer(self, permitted_value = for x in range nn.values)
        # Decoder layer for the layer optimization.
        decoder = Decoder(self, value = for x in permitted_values)

        # Calculating attention mechanism.
        attention  = MultiHeadAttentionBlock(Attention = self.attention, layer = self.LayerNormalization, Accounts = self.Accounts)

        position = PositionalEncoding(Attention = self.attention, layers = LayerNormalization)

        # For decoder layer we have multiple layers available in which we run multiple loops to reproduce the outputs for the procedures.

        decoder = position

        d1 = decoder(self, x, encoder_output)
        d1 = decoder(self, x, )
