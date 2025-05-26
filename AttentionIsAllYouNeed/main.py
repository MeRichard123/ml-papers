from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .Transformer import Transformer
from .Data import TranslationDataset, create_toy_dataset
from .train import train_transformer, evaluate_transformer, translate

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    eng_sentences, fr_sentences, src_vocab, tgt_vocab, src_idx2word, tgt_idx2word = create_toy_dataset()
    
    # Create train and validation datasets
    train_size = int(0.8 * len(eng_sentences))
    train_dataset = TranslationDataset(
        eng_sentences[:train_size], 
        fr_sentences[:train_size], 
        src_vocab, 
        tgt_vocab
    )
    val_dataset = TranslationDataset(
        eng_sentences[train_size:], 
        fr_sentences[train_size:], 
        src_vocab, 
        tgt_vocab
    )
    
    # Create data loaders
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    
    # Use smaller model for toy dataset
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,
        num_heads=2,
        d_ff=128,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab[''])
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_transformer(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_transformer(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('transformer_loss.png')
    plt.show()
    
    # Test translation
    test_sentences = [
        'hello how are you',
        'i love programming',
        'thank you very much'
    ]
    
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    
    print("\nTest Translations:")
    for sentence in test_sentences:
        translation = translate(model, sentence, src_vocab, tgt_idx2word, device)
        print(f"English: {sentence}")
        print(f"French: {translation}")
        print()

if __name__ == "__main__":
    main()