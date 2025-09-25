import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # dimension of the model, creating some dimension for the embedding layer.
        self.d_model = d_model
        # embedding layer initializing embedding with certain value.
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding_layer(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        dropout = nn.Dropout(dropout)

        # Empty position vector for storing position values.
        # shape: (seq_len, d_model).
        pe = torch.zeros(seq_len, d_model) # [[],[],[]].......

        # create a tensor where the range is from 0 to sequence length.
        position = torch.arange(0, seq_len).unsqueeze(1) # [0,1,2,3,4,5,6]

        # division term for the positional encoding.
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000)/d_model) #[e(), e(),......]

        number = position * div_term # here 

        pe[:, 0::2] = torch.sin(number)

        pe[:, 1::3] = torch.cos(number)

        # Changed the dimension of the vector to (1, seq_len, d_model) size
        pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add Input tensor X and the   
        return x + self.pe[:, :x.shape(1), :].requires_grad_(False)
    

    












    

