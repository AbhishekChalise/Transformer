import torch 
from models.embeddings import Embeddings, PositionalEncoding 
from models.attention import  MultiHeadAttentionBlock

def run_embedding_and_attention_tests():
    """
    A minimal test to check if embeddings and attention work together.
    We use fewer hyperparameters to keep it simple. 
    """

    print("--- Running Simple test: Embedding -> Attention ---")

    d_model = 32
    h = 4

    # ----- 2. Create Fake data --------
    # A single sentence with 10 words
    # Shape = (batch_size = 1, sequence_length = 10)

    input_tokens_ids = torch.randint(1, 100, (1,10))

    print(f"Start: Input shape is {input_tokens_ids}")

    try:

        token_emb = Embeddings(d_model = d_model, vocab_size = 100)
        token_vectors= token_emb.forward(input_tokens_ids)
        
        print(f"After token embedding the shape is: {token_vectors.shape}")
        
        pos_enc = PositionalEncoding(d_model = d_model, seq_len = 50, dropout=0)
        position_vectors = pos_enc.forward(input_tokens_ids)

        print(f"This is the shape after Positional Encoding: {position_vectors}")

        multi_head = MultiHeadAttentionBlock(d_model = d_model, h = h, dropout=0)

        # now we need to call its forward method for calculating the Key, Query and Value
        attention_output = multi_head.forward(query = position_vectors, key = position_vectors, value = position_vectors, mask = None)
        print(f"Step 3: After MultiHeadAttention, shape is {attention_output.shape}\n")

        # --- 4. final check ----
        print("---Completed test---")

        if position_vectors.shape == attention_output.shape:
            print("The output shape is correct. The modules are working correctly")
        else:
            print("Failure!, The shapes does not match")
        
    except Exception as e:
        print(f"The error is: {e}")

if __name__ == "__main__":
    run_embedding_and_attention_tests()