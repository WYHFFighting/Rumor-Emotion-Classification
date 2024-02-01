# -*- coding: utf-8 -*-
from sklearn import svm
from xgboost import XGBModel, XGBClassifier
import pandas as pd
import numpy as np
from transformers import logging, AutoTokenizer, AutoModel
from config import get_config
from data2 import load_dataset
import wandb
from sklearn.metrics import f1_score, recall_score, accuracy_score
import torch
import random
import argparse
import time
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MLModels:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-tokenizer', add_prefix_space = True)
        if args.model_name == 'svm':
            self.model = svm.SVC()
        elif args.model_name == 'xgboost':
            # self.model = XGBModel()
            self.model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic')

    def run(self):
        train_dataloader, test_dataloader = load_dataset(tokenizer = self.tokenizer,
                                                         train_batch_size = self.args.train_batch_size,
                                                         test_batch_size = self.args.test_batch_size,
                                                         val_batch_size = self.args.test_batch_size,
                                                         model_name = self.args.model_name,
                                                         method_name = self.args.method_name,
                                                         workers = self.args.workers,
                                                         tok = False)
        for inputs, targets, emotion, ddp_features in train_dataloader:
            main_fea = inputs['input_ids'].numpy()
            emotion = emotion.reshape(-1, 1).numpy()
            ddp_features = torch.mean(ddp_features, dim = 1).numpy()
            ff = np.concatenate([main_fea, emotion, ddp_features], axis = 1)
            targets = targets.numpy()
            self.model.fit(ff, targets)

        for inputs, targets, emotion, ddp_features in test_dataloader:
            main_fea = inputs['input_ids'].numpy()
            emotion = emotion.reshape(-1, 1).numpy()
            ddp_features = torch.mean(ddp_features, dim = 1).numpy()
            ff = np.concatenate([main_fea, emotion, ddp_features], axis = 1)
            targets = targets.numpy().astype('float32')
            predictions = self.model.predict(ff)

            acc = accuracy_score(targets, predictions)
            f1 = f1_score(targets, predictions)
            recall = recall_score(targets, predictions)

            print(f'{self.args.model_name}')
            print('acc:', acc)
            print('f1:', f1)
            print('recall:', recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', type = str, default = 'none',
                        choices = ['none', 'gru', 'rnn', 'bilstm', 'lstm', 'fnn', 'textcnn', 'attention',
                                   'lstm+textcnn',
                                   'lstm_textcnn_attention', 'BiLstm_attention'])  # 朴素贝叶斯

    '''Optimization'''
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--weight_decay', type = float, default = 0.01)

    '''Environment'''
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--backend', default = False, action = 'store_true')
    parser.add_argument('--workers', type = int, default = 0)
    parser.add_argument('--timestamp', type = int, default = '{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    parser.add_argument('--wandb', type = bool, default = False)
    parser.add_argument('--gram', type = int, default = 10)
    parser.add_argument('--optimizer', type = str, default = 'adam')

    parser.add_argument('--num_classes', default = 2)
    parser.add_argument('--train_batch_size', type = int, default = 8241)
    parser.add_argument('--test_batch_size', type = int, default = 2061)
    parser.add_argument('--val_batch_size', type = int, default = 2061)
    parser.add_argument('--num_epoch', type = int, default = 30)
    parser.add_argument('--model_name', type = str, default = 'xgboost')
    args = parser.parse_args()

    m = MLModels(args)
    m.run()

'''
svm
acc: 0.8665696263949539
f1: 0.8688602765855985
recall: 0.8659695817490495
'''