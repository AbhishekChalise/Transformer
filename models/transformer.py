import torch
import torch.nn as nn

from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from models.decoder import Projection_Layer
class Transformer(nn.Module):
    def __init__(self, encoder: EncoderLayer, decoder: DecoderLayer, projection_layer: Projection_Layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def create_masks(self, src, tgt):
        # Create all the masks needed for the encoder and decoder
        # This is a placeholder for the actual mask creation logic
        # src_mask prevents attention from focusing on <pad> tokens
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # Example mask
        # tgt_mask is the look-ahead mask + a padding mask
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # Example padding mask
        seq_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).bool()
        tgt_mask = tgt_mask & ~look_ahead_mask # Combine padding and look-ahead
        return src_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # This defines the complete translation workflow.

        # 1. Create the masks for the source and target sentences.
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # 2. Run the source sentence through the encoder.
        # The encoder reads the entire English sentence and creates the 'memory'.
        memory = self.encoder(src, src_mask)
        
        # 3. Run the target sentence and the encoder's memory through the decoder.
        # The decoder takes the encoder's knowledge and the French sentence so far.
        decoder_output = self.decoder(tgt, memory, src_mask, tgt_mask)
        
        # 4. Project the decoder's output into vocabulary scores.
        # This is the final step to get our word predictions.
        logits = self.projection_layer(decoder_output)
        
        return logits