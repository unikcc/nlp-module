#!/usr/bin/env python


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse

import yaml

from transformers import Trainer, TrainingArguments, IntervalStrategy

from attrdict import AttrDict

from src.tools import update_config, set_seed, compute_metrics
from src.loader import make_supervised_data_module
import transformers
from src.model import TextClassification
from loguru import logger



class Template:
    def __init__(self):
        config_file = 'src/config.yaml'

        config = AttrDict(yaml.load(open(config_file, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        config = update_config(config)

        set_seed(config.seed)
        self.config = config

    def forward(self):

        training_args = TrainingArguments(output_dir='./results', num_train_epochs=2, per_device_train_batch_size=16,
                                          logging_dir='./logs', evaluation_strategy=IntervalStrategy.EPOCH,
                                          per_device_eval_batch_size=16, warmup_steps=500, weight_decay=0.01, report_to=[])

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.bert_path, padding_side="right",use_fast=False)

        self.model = TextClassification(self.config, tokenizer)
        data_module, test_set = make_supervised_data_module(tokenizer, self.config)

        # Initialize trainer
        trainer = Trainer(model=self.model, args=training_args, compute_metrics=compute_metrics, **data_module)
        trainer.train()
        logger.info('Training finished, start evaluating')
        res = trainer.evaluate(test_set)
        print(res)


if __name__ == '__main__':
    template = Template()
    template.forward()
