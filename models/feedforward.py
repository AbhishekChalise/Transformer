import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model: int, d_iff: int, dropout_f: float):
        super().__init__()

        self.layer1 = nn.linear(d_model, d_iff)

        self.layer2 = nn.linear(d_iff, d_model)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_f)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass for the feed forward layer
        This inputs the layer with dimention (batch_size, seq_len, d_model)

        and also outputs the layer with dimention (batch_size, seq_len, d_model)
        """

        # This converts the size (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_iff)
        x = self.layer1(x)

        # Implement relu activation function
        x = self.activation(x)

        # Dropout the layers so that models learns efficiently
        x = self.dropout(x)

        # Next therefore convert the size to original (batch_size, seq_len, d_iff) -> (batch_size, seq_len, d_model)
        x = self.layer2(x)

