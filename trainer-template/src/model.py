#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from transformers import AutoModel

class TextClassification(nn.Module):
    def __init__(self, cfg, tokenizer):
        super(TextClassification, self).__init__()
        self.cfg = cfg 
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        self.tokenizer = tokenizer

        self.linear = nn.Linear(self.bert.config.hidden_size, cfg['num_classes'])
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, input_ids, input_masks, labels):
        res = self.bert(input_ids, attention_mask=input_masks)[1]
        output = self.linear(res)
        loss = nn.CrossEntropyLoss()(output, labels)
        res = { 'loss': loss, 'logits': output}
        return  res
