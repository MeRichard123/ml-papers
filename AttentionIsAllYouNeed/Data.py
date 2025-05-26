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
        src_indices = [self.src_vocab.get(word, self.src_vocab['']) for word in self.src_sentences[idx].split()]
        src_indices = [self.src_vocab['']] + src_indices + [self.src_vocab['']]
        
        # Convert target sentence to indices
        tgt_indices = [self.tgt_vocab.get(word, self.tgt_vocab['']) for word in self.tgt_sentences[idx].split()]
        tgt_indices = [self.tgt_vocab['']] + tgt_indices + [self.tgt_vocab['']]
        
        # Pad sequences
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]
        
        src_indices = src_indices + [self.src_vocab['']] * (self.max_len - len(src_indices))
        tgt_indices = tgt_indices + [self.tgt_vocab['']] * (self.max_len - len(tgt_indices))
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices[:-1], dtype=torch.long), # Input to decoder
            'tgt_y': torch.tensor(tgt_indices[1:], dtype=torch.long) # Expected output
        }
    
def create_toy_dataset():
    # Simple English to French translation pairs
    eng_sentences = [
        'hello how are you',
        'i am fine thank you',
        'what is your name',
        'my name is john',
        'where do you live',
        'i live in new york',
        'i love programming',
        'this is a test',
        'please translate this',
        'thank you very much'
    ]
    
    fr_sentences = [
        'bonjour comment vas tu',
        'je vais bien merci',
        'quel est ton nom',
        'je m appelle john',
        'où habites tu',
        'j habite à new york',
        'j aime programmer',
        'c est un test',
        's il te plaît traduis cela',
        'merci beaucoup'
    ]
    
    # Create vocabularies
    src_vocab = {'': 0, '': 1, '': 2, '': 3}
    tgt_vocab = {'': 0, '': 1, '': 2, '': 3}
    
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