from .Utils import create_padding_mask, create_look_ahead_mask
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