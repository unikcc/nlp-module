#!/use/bin/env python


"""
@Filanme: public.py
@Author:  unikcc
@Contact: libobo.uk@gmail.com
@Date:    2022/11/28 13:24:34 
"""


import os
import re
import random

import torch
import spacy
import numpy as np

from collections import Counter

from torch.utils.data import Dataset, DataLoader


class BaseDataloader:
    def __init__(self, config):
        self.config = config
        path = os.path.join(self.config.preprocessed_dir, '{}.pkl'.format(self.config.model_name))
    
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


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data) 


class BaseProcess:
    def __init__(self, config):
        self.spacy = spacy.load('en_core_web_sm') #加载spacy
        self.config = config 

    def clean_str(self, string):
        """
        Tokenization/string cleaning for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def read_file(self, filename):
        a = open(filename, 'r', encoding='utf-8')
        res = []
        for line in a:
            label, text = int(line[0]), self.clean_str(line[1:]).split()
            res.append((text, label))
        return res
    
    def normalize_word(self, word):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word
    
    def build_word_dict(self, dataset):
        dataset = [w for line in dataset.values() for w in line]
        word_list = []
        for text, _ in dataset:
            word_list += text
        word2id = {'pad': 0, 'unk': 1}
        word_count = Counter(word_list).most_common()
        for w, i in word_count:
            if i >= self.config.min_freq:
                word2id[w] = len(word2id)
        self.word_dict = word2id
    
    def forward(self):
        pass

    def forward(self):
        pass
