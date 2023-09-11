#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@filename: main.py
@dateTime: 2023-09-11 15:03:59
@author:   unikcc
"""

import argparse

import os
import nni
import yaml
import torch
import torch.nn as nn


from attrdict import AttrDict
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

from src.tools import update_config, set_seed, load_params, format_final
from src.model import TextCNN 
from src.loader import MyDataLoader

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Template:
    def __init__(self, args):
        config_file = 'src/config.yaml'

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
        self.re_init()
        self.scores = []
        self.lines = []
        self.save_name = 'src/{}.pth.tar'
    
    def re_init(self):
        self.preds = defaultdict(list)
        self.golds = defaultdict(list)
        self.keys = ['default']
    
    def add_instance(self, score, res):
        self.scores.append(score)
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax(self.scores)
        res = self.lines[best_id]
        return res
    
    def train(self):
        best_score, best_iter, best_res = 0, -1, ''
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            score, res = self.evaluate_step()
            self.re_init()

            self.add_instance(score, res)

            res = self.get_best()

            if score > best_score:
                if best_iter > -1:
                    os.system('rm -rf {}'.format(save_name))
                best_score, best_iter = score, epoch
                nni.report_intermediate_result(best_score)
                save_name = self.save_name.format(epoch)

                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)

                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                        save_name)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)
        
        score, res = self.final_evaluate(best_iter)

        save_name = self.save_name.format(epoch)
        os.system('rm -rf {}'.format(save_name))

        self.final_score, self.final_res = score, res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader)

        losses = []
        for i, data in enumerate(train_data):
            loss, _ = self.model(**data)
            loss.backward()
            losses.append(loss.item())

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.optimizer.step()
            self.model.zero_grad()

            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)

    def evaluate_step(self, dataLoader=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in enumerate(dataiter):
            with torch.no_grad():
                loss, output = self.model(**data)
                self.add_output(data, output)
        
        score, result = self.report_score()
        
        return score, result 
    
    def final_evaluate(self, epoch=0):
        PATH = self.save_name.format(epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        s, res = self.evaluate_step(self.test_loader)

        return s, res
    
    def add_output(self, data, output):
        data = data['input_labels']
        for i, key in enumerate(self.keys):
            self.preds[key] += output.argmax(-1).tolist()
            self.golds[key] += data.tolist()
    
    def report_score(self):
        res = []
        for i, key in enumerate(self.keys):
            acc = accuracy_score(self.golds[key], self.preds[key])

            if i == 0:
                p, r, f, report = precision_recall_fscore_support(self.golds[key], self.preds[key], average='binary')
            else:
                p, r, f, report = precision_recall_fscore_support(self.golds[key], self.preds[key], average='binary')
            line = [acc, p, r, f, report]
            res.append(line)
        return res[0][0], res

    def forward(self):

        (self.train_loader, self.valid_loader, self.test_loader), self.config = MyDataLoader(self.config).get_data()
      
        reports=  []
        self.model = TextCNN(self.config).to(self.config.device)

        self.config = load_params(self.config, self.model, self.train_loader)

        self.train()
        scores, line = self.final_score, self.final_res

        reports = [w[:4] for w in line]
        df = pd.DataFrame(reports, columns = ['acc','p','r', 'f'], index = ['sentiment'])
        result = format_final(df)

        nni.report_final_result(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--cuda_index', default=0)
    parser.add_argument('-pre', '--preprocess', default='yes')
    args = parser.parse_args()
    template = Template(args)
    template.forward()