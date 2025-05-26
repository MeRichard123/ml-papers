from torch import nn
from .Attention import MultiHeadAttention
from .Utils import PositionalEncoding, MLP

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):  
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.mlp = MLP(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with mask + residual connection + layer normalization
        self_attn_output = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-attention with mask + residual connection + layer normalization
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, attn_mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward network with residual connection and layer normalization
        ff_output = self.mlp(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # apply embedding and positional encoding
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)

        return x