#!/usr/bin/env python

"""
@Filanme: main.py
@Author:  unikcc
@Contact: libobo.uk@gmail.com
@Date:    2022/11/28 10:59:59 
"""

import argparse

import nni
import yaml
import torch


from attrdict import AttrDict
from loguru import logger
import numpy as np
import pandas as pd

from utils.tools import update_config, set_seed, load_params, load_params_bert, format_final
from utils.load_first import get_model_loader


class Template:
    def __init__(self, args):
        config_file = 'configs/{}_config.yaml'.format(args.model_name)

        self.model_func, self.load_func, self.train_func = get_model_loader(args.model_name)

        config = AttrDict(yaml.load(open(config_file, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        config = update_config(config)

        params = nni.get_next_parameter()
        names = []
        for k, v in vars(args).items():
            setattr(config, k, v)
        for k, v in params.items():
            config[k] = v
            names.append(v)

        set_seed(config.seed)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_name] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        self.config = config

    def forward(self):
        (self.trainLoader, self.validLoader, self.testLoader), self.config = self.load_func(self.config).get_data()
      
        reports=  []
        self.model = self.model_func(config=self.config).to(self.config.device)

        self.config = load_params(self.config, self.model, self.trainLoader)

        trainer = self.train_func(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        trainer.train()
        scores, line = trainer.final_score, trainer.final_res

        reports = [w[:4] for w in line]
        df = pd.DataFrame(reports, columns = ['acc','p','r', 'f'], index = ['sentiment'])
        result = format_final(df)

        nni.report_final_result(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cdd', '--uda_index', default=2)
    parser.add_argument('-pre', '--preprocess', default='yes')
    parser.add_argument('-md', '--model_name', \
                        default='textcnn',
                        choices=['bert', 'textcnn', 'mtl', 'testlc', 'fasttext'])
    args = parser.parse_args()
    template = Template(args)
    template.forward()