from torch.utils.data import Dataset
from typing import List
from transformers import AutoTokenizer
import torch

class IMDBDataset(Dataset):
    """Dataset with strings to predict pos/neg
    Args:
        text (List[str]): list of strings
        label (List[str]): list of corresponding labels (spam/ham)
        data_size (int): number of data rows to use
    """


    def __init__(
            self,
            text: List[str],
            label: List[str],
            tokenizer: AutoTokenizer,
            max_length: int = 64,
            data_size: int = 1000):
        
        if data_size > len(text):
            raise ValueError(f"Maximum rows in dataset {len(text)}")
        self.text = text[:data_size]
        self.label = label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.text[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
            
        return (item, self.label[idx])