#!/use/bin/env python

"""
@Filanme: engine.py
@Author:  unikcc
@Contact: libobo.uk@gmail.com
@Date:    2022/11/28 15:13:08 
"""

import os

import torch
import nni

import numpy as np
import torch.nn as nn

from tqdm import tqdm

class LCTrainer(object):
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''

        self.scores = []
        self.lines = []

        self.re_init()
    
    def re_init(self):
        pass

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
            # if i > 80: break
            loss, _ = self.model(**data)
            loss.backward()
            losses.append(loss.item())

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.optimizer.step()
            # self.config.scheduler.step()
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
        pass

    def report_score(self):
        pass

    def add_instance(self, score, res):
        self.scores.append(score)
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax(self.scores)
        res = self.lines[best_id]
        return res