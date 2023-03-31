#!/use/bin/env python


"""
@Filanme: tools.py
@Author:  unikcc
@Contact: libobo.uk@gmail.com
@Date:    2022/11/14 13:26:57 
"""

import os
import random


import nni
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.optim import AdamW


def update_config(config):

    dirs = ['preprocessed_dir', 'target_dir', 'dataset_dir']
    for dirname in dirs:
        if dirname in config:
            config[dirname] = os.path.join(config.data_dir, config[dirname])
    
    return config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_params(config, model, fold_data):
    if any(w in config.model_name for w in ['mtl', 'bert']):
        return load_params_bert(config, model, fold_data)
    
    if 'cnn' in config.model_name:
        return load_cnn(config, model, fold_data)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                               lr=float(config.learning_rate),
                               eps=float(config.adam_epsilon),
                               weight_decay=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.epoch_size * fold_data.__len__())

    config.score_manager = ScoreManager()

    config.optimizer = optimizer
    config.scheduler = scheduler

    return config

def load_cnn(config, model, fold_data):
    config.optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=float(config['learning_rate']))
    config.score_manager = ScoreManager()
    return config

def load_params_bert(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = set(model.bert.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': float(config.bert_learning_rate),
            'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': float(config.bert_learning_rate),
            'weight_decay': 0.0},
        {'params': other_params,
            'lr': float(config.learning_rate),
            'weight_decay': float(config.weight_decay)},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                            eps=float(config.adam_epsilon))

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.epoch_size * fold_data.__len__())

    config.score_manager = ScoreManager()

    config.optimizer = optimizer
    config.scheduler = scheduler

    return config


class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []

    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)

    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return res

def report_score(golds, preds):
    def my_report(golds, preds):
        tp, fn, fp = 0, 0, 0
        # fn = sum(golds)
        # fp = sum(preds)
        # tp = len([1 for w, z in zip(golds, preds) if w == z == 1])

        # p = tp / fp if fp > 0 else 0
        # r = tp / fn if fn > 0 else 0
        # f = 2 * p * r / (p + r) if p + r > 0 else 0

        if isinstance(golds[0], int):
            fn = len(golds)
            fp = len(preds)
            tp = len([1 for w, z in zip(golds, preds) if w == z])
            acc = accuracy_score(golds, preds)
            p, r, f, report = precision_recall_fscore_support(golds, preds, average='weighted')
        else:
            p, r, f, report = precision_recall_fscore_support(golds, preds, average='micro')
            gd = [w for line in golds for w in line]
            pd = [w for line in preds for w in line]
            fn = len(gd)
            fp = len(pd)
            tp = len([1 for w, z in zip(gd, pd) if w == z])
            acc = accuracy_score(gd, pd)
        # p, r, f, report = precision_recall_fscore_support(golds, preds, average='micro')

        str_length, num_length = 10, 10

        head_line = '{:>' + str(str_length) + '}'
        content_line_f = '{:>' + str(num_length) + '.4f}'
        content_line_i = '{:>' + str(num_length) + '}'
        res = [[head_line.format(w) for w in 'acc p r f support pred gold'.split()]]
        content0 = [content_line_f.format(w) for w in [acc, p, r, f]]
        content1 = [content_line_i.format(w) for w in [tp, fp, fn]]
        res.append(content0 + content1)
        res = [' '.join(w) for w in res]
        res = '\n'.join(res)
        return acc, p, r, f, res

    if isinstance(golds, list):
        return my_report(golds, preds)

    res = []
    for k, v in golds.items():
        line = my_report(golds[k], preds[k])
        res.append(line)
    return res

def get_save_name(config, epoch):
    save_name = '{}_{}_{}.pth.tar'.format(config.model_name, config.seed, epoch)
    save_name = os.path.join(config.target_dir, save_name)
    return save_name 



def init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight) # random 
        nn.init.constant_(module.bias.data, 0.0) # 0.0-> 85.45, 0.1-> 85.28
    elif isinstance(module, nn.Conv2d):
        nn.init.uniform_(module.weight.data, -0.1, 0.1) # 81.71
        nn.init.constant_(module.bias.data, 0.0) # 无所谓
    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1) # 81.71


def format_final(df):
    rows = list(df.index)
    cols = list(df.columns)
    res = {}
    for i, line in enumerate(df.to_numpy()):
        for j, num in enumerate(line):
            res['{}_{}'.format(rows[i], cols[j])] = num
    
    res['default'] = res['sentiment_f']
    return res

