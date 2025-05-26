from torch import nn
import torch

class PositionalEncoding(nn.Module): # RoPE
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Apply sine to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Register pe as a buffer so it is not a model parameter but still part of the model's state
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
    


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def create_padding_mask(seq, pad_idx=0):
    # Create a mask for padding tokens: 1 for non-pad, 0 for pad
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    # Create a triangular mask to hide future tokens
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask