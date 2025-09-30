import torch
import torch.nn as nn
import math

# Import all the building blocks

from .embeddings import Embeddings, PositionalEncoding
from .encoder import EncoderLayer
from .decoder import Decoder

class ProjectionLayer(nn.Module):
    """
    Final layer to project output to the vocabulary size.
    This produces the raw scores (logits) for each vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the ProjectionLayer

        Args:
        d_models(int): The dimensinality of the vector
        vocba_size(int): The size of target vocabulary
        """

        super().__init__()
        # A simple linear layer

        linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the projection layer 
        """


class Transformer(nn.Module):
    


        