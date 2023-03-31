#!/use/bin/env python


"""
@Filanme: train.py
@Author:  unikcc
@Contact: libobo.uk@gmail.com
@Date:    2022/11/14 13:38:38 
"""

from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from trainer.engine import LCTrainer


class textCNNTrainer(LCTrainer):
    def re_init(self):
        self.preds = defaultdict(list)
        self.golds = defaultdict(list)
        self.keys = ['default']
    
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

