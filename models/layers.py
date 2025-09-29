import torch
import torch.nn as nn
import math

class LayerNormalization():

    def __init__(self, d_model, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        # This is the alpha and beta which causes mean to be zero and variance to be 1.
        self.alpha = nn.Parameter(torch.ones(d_model))
        
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):

        # Calaulate mean value 
        mean = x.mean(dtype = float, dim = -1, keepdim = True)

        std = x.float().std(dim = -1, keepdim = True)

        normalized_x = (x - mean) / (std + self.eps)

        return self.alpha * normalized_x + self.beta

        





