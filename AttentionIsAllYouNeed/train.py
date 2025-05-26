from Utils import create_padding_mask, create_look_ahead_mask
import torch

def train_transformer(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_y = batch['tgt_y'].to(device)
        
        # Create masks
        src_mask = create_padding_mask(src)
        tgt_mask = create_padding_mask(tgt) & create_look_ahead_mask(tgt.size(1)).to(device)
        
        # Forward pass
        output = model(src, tgt, src_mask, tgt_mask)
        
        # Reshape output and target for loss calculation
        output = output.view(-1, output.size(-1))
        tgt_y = tgt_y.view(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


def evaluate_transformer(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_y = batch['tgt_y'].to(device)
            
            # Create masks
            src_mask = create_padding_mask(src)
            tgt_mask = create_padding_mask(tgt) & create_look_ahead_mask(tgt.size(1)).to(device)
            
            # Forward pass
            output = model(src, tgt, src_mask, tgt_mask)
            
            # Reshape output and target for loss calculation
            output = output.view(-1, output.size(-1))
            tgt_y = tgt_y.view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_y)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(val_loader)


def translate(model, sentence, src_vocab, tgt_idx2word, device, max_len=100):
    model.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.split()
    src_indices = [src_vocab.get(word, src_vocab['<unk>']) for word in tokens]
    src_indices = [src_vocab['<sos>']] + src_indices + [src_vocab['<eos>']]
    
    # Pad source sequence
    src_indices = src_indices + [src_vocab['<pad>']] * (max_len - len(src_indices))
    src_indices = src_indices[:max_len]
    
    # Convert to tensor
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
    
    # Create mask
    src_mask = create_padding_mask(src_tensor)
    
    # Get encoder output
    enc_output = model.encode(src_tensor, src_mask)
    
    # Initialize decoder input with  token
    dec_input = torch.tensor([[src_vocab['<sos>']]], dtype=torch.long).to(device)
    
    # Generate translation
    output_indices = []
    
    for _ in range(max_len):
        # Create mask for decoder input
        tgt_mask = create_look_ahead_mask(dec_input.size(1)).to(device)
        
        # Get decoder output
        dec_output = model.decode(dec_input, enc_output, src_mask, tgt_mask)
        
        # Get predicted token
        pred = model.linear(dec_output[:, -1])
        pred_idx = pred.argmax(dim=-1).item()
        
        # Add predicted token to output
        output_indices.append(pred_idx)
        
        # Check if end of sequence
        if pred_idx == src_vocab['<eos>']:
            break
        
        # Update decoder input
        dec_input = torch.cat([dec_input, torch.tensor([[pred_idx]], dtype=torch.long).to(device)], dim=1)
    
    # Convert indices to words
    output_words = [tgt_idx2word.get(idx, '<unk>') for idx in output_indices]
    
    # Remove special tokens
    output_words = [word for word in output_words if word not in ['<pad>', '<sos>', '<eos>', '<unk>']]
    
    return ' '.join(output_words)