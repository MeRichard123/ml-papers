import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # Convert source sentence to indices
        src_tokens = self.src_sentences[idx].split()
        src_indices = [self.src_vocab['<sos>']]  # Start with <sos>
        src_indices.extend([self.src_vocab.get(word, self.src_vocab['<unk>']) for word in src_tokens])
        src_indices.append(self.src_vocab['<eos>'])  # End with <eos>
        
        # Convert target sentence to indices
        tgt_tokens = self.tgt_sentences[idx].split()
        tgt_indices = [self.tgt_vocab['<sos>']]  # Start with <sos>
        tgt_indices.extend([self.tgt_vocab.get(word, self.tgt_vocab['<unk>']) for word in tgt_tokens])
        tgt_indices.append(self.tgt_vocab['<eos>'])  # End with <eos>
        
        # Pad sequences
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]
        
        src_indices = src_indices + [self.src_vocab['<pad>']] * (self.max_len - len(src_indices))
        tgt_indices = tgt_indices + [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_indices))
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices[:-1], dtype=torch.long),  # Input to decoder
            'tgt_y': torch.tensor(tgt_indices[1:], dtype=torch.long)  # Expected output
        }

def sanitise(text):
    import re
    import unicodedata

    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.category(c) == 'Mn'])
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text

def create_toy_dataset():
    # Simple English to French translation pairs
    eng_sentences = []
    fr_sentences = []
    with open('AttentionIsAllYouNeed\\fra.txt', 'r', encoding='utf-8') as f:
        pairs = f.readlines()

        for pair in pairs:
            eng, fr = pair.split('\t')
            eng_sentences.append(sanitise(eng))
            fr_sentences.append(sanitise(fr))

    # Create vocabularies with special tokens
    src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # Add words to vocabularies
    i = 4
    for sent in eng_sentences:
        for word in sent.split():
            if word not in src_vocab:
                src_vocab[word] = i
                i += 1
    
    i = 4
    for sent in fr_sentences:
        for word in sent.split():
            if word not in tgt_vocab:
                tgt_vocab[word] = i
                i += 1
    
    # Create reverse vocabularies for decoding
    src_idx2word = {idx: word for word, idx in src_vocab.items()}
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
    
    return eng_sentences, fr_sentences, src_vocab, tgt_vocab, src_idx2word, tgt_idx2word