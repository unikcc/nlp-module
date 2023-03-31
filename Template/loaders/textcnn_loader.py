#!/use/bin/env python


"""
@Filanme: utils.py
@Author:  unikcc
@Contact: libobo.uk@gmail.com
@Date:    2022/11/14 13:39:04 
"""

import os
import torch
from .public import BaseProcess, BaseDataloader

class MyDataLoader(BaseDataloader):
    def __init__(self, config):
        config.preprocessor = Preprocessor(config)
        super().__init__(config)
    
    def collate_fn(self, data):

        input_ids, input_labels = zip(*data)

        max_lengths = max(map(len, input_ids))
        padding  = lambda x: [(w + [0] * (max_lengths - len(w)))[:max_lengths] for w in x]
        input_ids = padding(input_ids)

        res = {
            'input_ids': input_ids, 
            'input_labels': input_labels,
        }
        noused = []
        res = {k : torch.tensor(v).to(self.config.device) for k, v in res.items()}
        return res

class Preprocessor(BaseProcess):
    def __init__(self, config=None):
        super().__init__(config)
    
    def transformer2indices(self, dataset):
        res = []
        for line, label in dataset:
            line = [self.word_dict[w if w in self.word_dict else 'unk'] for w in line]
            res.append((line, label))
        return res
    
    def forward(self):
        modes = 'train valid test'.split()
        dataset = {}
        for mode in modes:
            path = os.path.join(self.config.dataset_dir, 'stsa.binary.{}'.format(mode))
            dataset[mode] = self.read_file(path)
        
        self.build_word_dict(dataset)
        
        res = []
        for mode in modes:
            data = self.transformer2indices(dataset[mode])
            res.append(data)
        res.append(self.word_dict)
        
        return res

if __name__ == '__main__':
    template = Preprocessor()
    template.forward()
