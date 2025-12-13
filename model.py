import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # Check last line (just before table) of 3.4 (Embeddings and Softmax)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # To reduce overfitting (not relying on same neurons each time for giving output)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) -- for 1 sentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)) # Same expression as given in paper but more optimised version
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Shape = (1, seq_len, d_model) -- for batch of sentences

        self.register_buffer("pe", pe) # To save this tensor along with module and saying that this is not a parameter

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Do not record gradient of this positional encoding tensor
        # pe[:, :x.shape[1], :] = Take positional encodings up to the sequence length of x (of all batches) -- think with keeping in mind shape of pe
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps # Epsilon - used for numerical stability (to avoid division by 0 (in this case if sigma is 0 or near to zero (check formula)))
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(1)) # Additive parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # Mean along last dimension (d_model) and dont reduce the dimension (keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1 (b1 is internally defined in nn.Linear)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int , dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of each vector as seen by head
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo -- in ppr it is h*d_v,d_model [d_v==d_k and d_k*h=d_model]

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        # For understanding ignore (batch, h) and think like (seq_len, d_k) @ (d_k, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # Mask the elements having mask value = 0 with the value -10^9 (-∞)
        # softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Multiply by respective weight matrices
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q) # multiply by w_q i.e. q'(or query) = (w_q)^T @ q
        key = self.w_k(k)
        values = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # transpose creates a non-contiguous tensor (changes strides); contiguous() rearranges data in memory so view() can reshape it safely
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by w_o
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # Pre-LayerNorm residual connection: x + Dropout(sublayer(LN(x))) [sublayer = prev layer eg. attention/FFN]
        # Note: The ppr uses Post-LN: LN(x + Dropout(sublayer(x))), but Pre-LN is more stable and widely used in modern Transformers.
        return x + self.dropout(sublayer(self.norm(x)))