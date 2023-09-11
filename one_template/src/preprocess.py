#!/use/bin/env python


"""
@Filanme: public.py
@Author:  unikcc
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


class Preprocessor:
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
