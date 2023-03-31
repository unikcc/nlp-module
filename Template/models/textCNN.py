#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Bobo Li
@contact: 932974672@qq.com
@file: model.py
@time: 2020/12/12 13:51
"""

import torch
import torch.nn as nn
import numpy as np
from utils.tools import init_esim_weights


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(len(config.word_dict), config.emb_dim)
        # self.embeddings.weight.requires_grad = False

        filters = config['filters']
        self.cnn = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, config['output_channels'], [w, config.emb_dim]),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)) for w in filters])

        self.linear = nn.Linear(config['output_channels'] * len(filters), 2, bias=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.scale = np.sqrt(3.0 / config.emb_dim)
        self.apply(init_esim_weights)

    def forward(self, input_ids, input_labels):
        # input: (batch_size, sentence_length, emb_dim)
        input = self.embeddings(input_ids).unsqueeze(1)
        cnn = [conv(input) for conv in self.cnn]
        output = torch.cat(cnn, 1).squeeze(2).squeeze(2)
        output = self.dropout(output)
        output = self.linear(output)

        critrion = nn.CrossEntropyLoss()
        loss = critrion(output, input_labels)
        return loss, output
