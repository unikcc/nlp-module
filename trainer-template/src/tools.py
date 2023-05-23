#!/use/bin/env python


import os
import random

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': precision_recall_fscore_support(labels, predictions, average='weighted')[2],
    }
