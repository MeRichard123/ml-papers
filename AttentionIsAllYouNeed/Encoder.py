from torch import nn
from .Attention import MultiHeadAttention
from .Utils import PositionalEncoding, MLP

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.mlp = MLP(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output)) # Add residual connection

        # Feed-forward network with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply embedding and positional encoding
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x
        