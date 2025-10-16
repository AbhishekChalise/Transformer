import torch
import torch.nn as nn

from models.embeddings import PositionalEncoding
from models.embeddings import Embeddings
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

        # This is a standard neural network layer to process the information further.
        self.feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)

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
        # This is called Residual Connection 
        # if we calculate gradient then without residual the derivative may come very closer to zero
        # therefore if we add the tgt matrix and dropout then 

        tgt = self.linear_norm1(tgt)
        # Here we normalize the target values so that the target matrix is stable.

        # Step2: Cross - Attention
        # Goal: Connect the French Sentence to the original English sentences
        # the model asks: I have understood "Le chat"
        # Now what part of the cat is black i should focus on to decide the next word

        cross_attention_output = self.cross_attention_block(query = tgt, key = memory, value = memory, mask = src_mask)
        # WHY THIS LINE? This is the core of the translation.
        # WHY (query=tgt)? The `query` is the question. The French sentence so far ("Le chat")
        #   is used to form a question to ask the English sentence.
        # WHY (key=memory, value=memory)? The `memory` (from the encoder) acts as the knowledge base.
        #   The model compares its question (`query`) to the `key` (which represents the English words)
        #   to find the most relevant part. In this case, it will find that "Le chat" is highly
        #   related to "The cat". It then retrieves the information from that part using the `value`.

        tgt = tgt + self.dropout(cross_attention_output)
        # This is for residual connection so that we wont face the gradient descend issues.
        tgt = self.linear_norm2(tgt)
        # Step3: feed forward layer
        ff_output = self.feed_forward_block(tgt)

        tgt  = tgt + self.dropout(ff_output)
        # Why this line A residual Connection.

        tgt = self.linear_norm3(tgt)
        # the output of this will be input to the next decoder layer

        return tgt
    # this tgt is now more richer and context aware.


class Decoder(nn.Module):
    
    def __init__(self, tgt_vocab_size: int, d_model: int, num_layers: int, h: int, d_ff: int, dropout: float, seq_len: int):
        super().__init__()

        # Step1: Word Embeddings and Positional Encoding for the target (French) sentence.
        self.embedding = Embeddings(d_model, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

        # Step 2: Create a stack of N identical decoder layers.
        # WHY nn.ModuleList? It's a special list that correctly registers all the
        # layers with PyTorch so they are properly trained.
        self.layers = nn.ModuleList(
            [SimpleDecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # This forward pass defines the data flow for the entire decoder stack.

        # 1. Prepare the input target tensor.
        # It starts as token IDs, e.g., [56, 34, 9, 88]
        # After embedding and positional encoding, it's a rich tensor ready for processing.
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)

        # 2. Pass the data through the stack of layers.
        # The output of one layer becomes the input to the next.
        for layer in self.layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask) # Pass it through one DecoderLayer
        
        return tgt # The final output of the decoder stack

class Projection_Layer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # The output tensor (logits) will have a shape of (batch_size, seq_len, vocab_size).
        # For each word in the sequence, it gives a score for every word in the vocabulary.

        return self.proj(x)

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
