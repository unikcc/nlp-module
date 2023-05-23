#!/use/bin/env python


import os
import torch
from torch.utils.data import Dataset 
import transformers
import logging
import pickle as pkl
from typing import Dict
from dataclasses import dataclass 

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data, mode):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        cur_data = data[mode]
        self.input_ids = [w[0] for w in cur_data]
        self.labels = [w[1] for w in cur_data]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        x = (self.input_ids[i], self.labels[i])
        return x

@dataclass
class CollateFN:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        input_ids, labels = list(zip(*instances))
        input_ids = [' '.join(w) for w in input_ids]
        batch_input = self.tokenizer.batch_encode_plus(input_ids, return_tensors="pt", pad_to_max_length=True)
        IGNORE_INDEX = -100
        labels = list(labels)
        return {
            'input_ids': torch.tensor(batch_input['input_ids'], dtype=torch.long),
            'input_masks': torch.tensor(batch_input['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long), 
        }

class Preprocessor:
    def __init__(self, config=None):
        super().__init__(config)
    
    def transformer2indices(self, dataset):
        res = []
        for line, label in dataset:
            line = [self.word_dict[w if w in self.word_dict else 'unk'] for w in line]
            res.append((line, label))
        return res
    
    def read_file(self, filename):
        a = open(filename, 'r', encoding='utf-8')
        res = []
        for line in a:
            label, text = int(line[0]), self.clean_str(line[1:]).split()
            res.append((text, label))
        return res
    
    def forward(self):
        modes = 'train valid test'.split()
        dataset = {}
        for mode in modes:
            path = os.path.join(self.config.dataset_dir, 'stsa.binary.{}'.format(mode))
            dataset[mode] = self.read_file(path)
        
        return dataset


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, config) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    path = os.path.join(config.preprocessed_dir, '{}.pkl'.format(config.model_name))
    if not os.path.exists(path):
        preprocessor = Preprocessor(config)
        data = preprocessor.forward()
        with open(path, 'wb') as f:
            pkl.dump(data, f)
    else:
        with open(path, 'rb') as f:
            data = pkl.load(f)
    train_dataset = SupervisedDataset(data, 'train')
    valid_dataset = SupervisedDataset(data, 'valid')
    test_dataset = SupervisedDataset(data, 'test')
    data_collator = CollateFN(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator), test_dataset
