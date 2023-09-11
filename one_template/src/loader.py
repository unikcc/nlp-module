#!/use/bin/env python

#!/usr/bin/env python
# _*_ coding:utf-8 _*_


"""
@filename: loader.py
@dateTime: 2023-09-11 15:04:39
@author:   unikcc
"""

import os
import torch
import numpy as np
import random

from torch.utils.data import DataLoader, Dataset
from src.preprocess import Preprocessor

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data) 


class MyDataLoader():
    def __init__(self, config):
        config.preprocessor = Preprocessor(config)
        self.config = config
    
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
    
       
    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def get_data(self):
        self.data = self.config.preprocessor.forward()
        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]

        load_data = lambda dataset : DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
            shuffle=self.config.shuffle, batch_size=self.config.batch_size, collate_fn=self.collate_fn)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])


        res = [train_loader, valid_loader, test_loader]

        return res, self.config


