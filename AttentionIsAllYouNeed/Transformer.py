from torch import nn
from .Encoder import Encoder
from .Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                  d_ff=2048, num_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        enc_output = self.encoder(src, mask=src_mask)
        
        # Decode target sequence
        dec_output = self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)

        # Final linear layer to project to target vocabulary size
        output = self.linear(dec_output)
        return output
    
    def encode(self, src, src_mask=None):
        return self.encoder(src, mask=src_mask)
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        return self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)