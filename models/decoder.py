import torch
import torch.nn as nn

from models.attention import MultiHeadAttentionBlock
from models.layers import LayerNormalization
from models.feedforward import FeedForwardNetwork

class SimpleDecoderLayer(nn.Module):

    def __init__(self, d_model: int, h: int, d_ff: int, dropout:float):
        super().__init__()

        # ---Defining the layers components
        # these are the three core operations the decoder will perform in sequence.
        # This block will look at the French sentence generated so far ("Le  chat").
        self.self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)

        # This block will connect the French sentence to the English Sentence.
        self.cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)

        # These are the helper layers to make the training stable

        self.linear_norm1 = LayerNormalization(d_model)
        self.linear_norm2 = LayerNormalization(d_model)
        self.linear_norm3 = LayerNormalization(d_model)

        self.dropout  = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        This function describes the step-by-step process for generating the next part of the translation.

        Args:
        tgt: The French sentence generated so far (e.g., the vectors for "Le", "chat).
        memory: The Encoders output (the meaning of "The cat is black").
        src_mask: Mask for the English sentence(used to ignore the padding tokens).
        tgt_mask: Mask for the French Sentence(crucial for training).

        """

        #------------------------------------------------------------------------------------------------------------------------------
        # STEP 1: Masked Self-Attention
        # GOAL: Understand the context of the French sentence generated so far.
        # Before deciding on the next word, the model must understand "Le chat".

        # -----------------------------------------------------------------------------------------------------------------------------

        self_atten_out = self.self_attention_block(tgt, tgt, tgt, tgt_mask)
        # Why this line because we are feeding current French Sentence ('tgt') into attention block.
        # Why (tgt, tgt, tgt) ? Because this is self attention. The sentence is looking at itself to find relationship between its own words. For example,
        # it learns that chat is related to "Le"
        # Why (tgt_mask) it is because we are hiding this in order to  not make the model peek at the future tokens in order to make them sure that they wont cheat.



        tgt = tgt + self.dropout(self_atten_out)
        # why this becuase we dont want to loose the original tgt
        # This is called Residual Connection because
        # if we calculate gradient then without residual the derivative may come very closer to zero
        # therefore if we add the tgt matrix and dropout then 

        tgt = self.linear_norm1(tgt)
        # Here we normalize the target values so that the target matrix is stable.

        # Step2: Cross - Attention
        # Goal: Connect the French Sentence to the original English sentences
        # the model asks: I have understood the 