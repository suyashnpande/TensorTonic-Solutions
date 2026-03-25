import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_token = [
            self.pad_token, 
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        words = sorted(set(word for text in texts for word in text.split()))
        
        vocab = {token:i for i, token in enumerate(special_token)}
                
        for word in words:
            if word not in vocab:
                vocab[word]= len(vocab)
                
        self.word_to_id=vocab
        self.vocab_size = len(vocab)
        self.id_to_word = {i: token for token , i in vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        tokens=[]
        for word in text.split():
            if word not in self.word_to_id:
                tokens.append(self.word_to_id[self.unk_token])
            else:
                tokens.append(self.word_to_id[word])
                
        return tokens
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words= [
            self.id_to_word.get(i, self.unk_token) for i in ids
        ]
        return " ".join(words)
