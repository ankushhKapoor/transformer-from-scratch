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
        return self.embedding(x) * math.sqrt(self.d_model) # Check last line before table of 3.4 (Embeddings and Softmax)
    
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
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