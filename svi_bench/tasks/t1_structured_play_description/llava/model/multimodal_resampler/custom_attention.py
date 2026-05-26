import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask if provided (useful for masked self-attention in transformers)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to normalize scores, producing attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute the final output as weighted values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
    

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, input_size, intermediate_size, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        assert intermediate_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = intermediate_size // num_heads
        self.intermediate_size = intermediate_size

        # Linear layers for Q, K, V for all heads
        self.query = nn.Linear(input_size, intermediate_size)
        self.key = nn.Linear(input_size, intermediate_size)
        self.value = nn.Linear(input_size, intermediate_size)
        
        # Output linear layer
        self.out_proj = nn.Linear(intermediate_size, input_size)

    def forward(self, q, k, v, mask=None):
        N, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape
        _, seq_len_v, _ = v.shape
        assert seq_len_k == seq_len_v, "Key and Value sequences must have the same length"

        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)

        # Reshape Q, K, V to (N, num_heads, seq_len, head_dim)
        Q = Q.view(N, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform scaled dot-product attention and concatenate heads
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(N, seq_len_q, self.intermediate_size)

        # Final linear transformation
        return self.out_proj(out)