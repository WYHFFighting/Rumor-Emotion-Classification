import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel

from config import get_config
# dara/model
from dara import load_dataset
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, TextCNN_Model, Transformer_CNN_RNN, \
    Transformer_Attention,BiLstm_attention_Model, MLP

import wandb
from sklearn import metrics
import torch
from sklearn.metrics import f1_score, recall_score, precision_score

# from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM
# from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
# from transformers import Trainer, TrainingArguments, pipeline
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        # Create model
        if args.model_name == 'xlnet':
            self.tokenizer = AutoTokenizer.from_pretrained('chinese-xlnet-base-pytorch', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('chinese-xlnet-base-pytorch')
        elif args.model_name == 'xlnet_mid':
            self.tokenizer = AutoTokenizer.from_pretrained('chinese_xlnet_mid_pytorch', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('chinese_xlnet_mid_pytorch')
        elif args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('chinese-bert-wwm-ext', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('chinese-bert-wwm-ext')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-tokenizer', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('chinese-roberta')
        elif args.model_name is None:
            self.tokenizer = None
        else:
            raise ValueError('unknown model')
        # Operate the method
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(base_model, args.num_classes, self.input_size, args)
        elif args.method_name == 'none':
            self.Mymodel = MLP(base_model, args.num_classes, self.input_size, args)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'textcnn':
            self.Mymodel = TextCNN_Model(base_model, args.num_classes)
        elif args.method_name == 'attention':
            self.Mymodel = Transformer_Attention(base_model, args.num_classes)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        elif args.method_name == 'lstm_textcnn_attention':
            self.Mymodel = Transformer_Attention(base_model, args.num_classes)
        elif args.method_name == 'BiLstm_attention':
            self.Mymodel = BiLstm_attention_Model(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):

        '''train_loss, n_correct, n_train 分别初始化为0，用于记录训练过程中的累计损失、正确预测的数量和训练样本的总数量。'''
        train_loss, n_correct, n_train = 0, 0, 0
        prds_list = []
        labels_list = []
        # Turn on the train mode
        '''将模型切换到训练模式，这通常会启用具有不同行为的层，例如批标准化层和Dropout层。'''
        self.Mymodel.train()
        for inputs, targets, emotion, ddp_features in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            # 将数据移动到GPU上
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            emotions = emotion.to(self.args.device)
            ddp_features = ddp_features.to(self.args.device)
            # emotions = {k: v.to(self.args.device) for k, v in emotion.items()}
            # 预测emotions
            predicts = self.Mymodel(inputs, emotions, ddp_features)
            # 跟目标值进行匹配，计算损失值
            loss = criterion(predicts, targets)
            # 清空模型参数的梯度。在每次迭代开始时，
            # 需要将梯度置零，以便在下一次迭代中计算新的梯度。
            optimizer.zero_grad()
            # 执行反向传播，计算模型参数关于损失函数的梯度。
            # 通过调用这个方法，PyTorch会自动计算整个计算图上的梯度。
            loss.backward()
            # 该方法用于根据梯度更新模型参数。
            # 优化器（例如随机梯度下降优化器）使用损失函数的梯度来更新模型的权重，以最小化损失函数。
            optimizer.step()
            #  得到整个训练集的累计损失|将当前批次的损失乘以批次中样本的数量（targets.size(0)），并累加到train_loss中。
            train_loss += loss.item() * targets.size(0)
            # 计算当前批次中模型正确预测的样本数量，并累加到n_correct中。
            # 这里使用torch.argmax获取预测值中概率最大的类别，然后与目标标签进行比较。
            prds = torch.argmax(predicts, dim=1)
            n_correct += (prds == targets).sum().item()
            # 将当前批次中样本的数量累加到n_train中，用于后续计算训练准确率。
            n_train += targets.size(0)
            prds_list.append(prds)
            labels_list.append(targets)

        prds_all = torch.cat(prds_list)
        labels_all = torch.cat(labels_list)
        f1 = f1_score(labels_all.cpu().detach().numpy(), prds_all.cpu().detach().numpy(), average = 'macro')
        precise = precision_score(labels_all.cpu().detach().numpy(), prds_all.cpu().detach().numpy(), average = 'macro')
        recall = recall_score(labels_all.cpu().detach().numpy(), prds_all.cpu().detach().numpy(), average = 'macro')

        # f1 = f1_score(labels_all, prds_all)
        return train_loss / n_train, n_correct / n_train, f1, precise, recall

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        prds_list = []
        labels_list = []
        n_TP, n_FP, n_FN,x,x1,x0=0,0,0,0,0,0
        n_TP_0,x1_0,x0_0=0,0,0
        n_TP_2, x1_2, x0_2 = 0, 0, 0
        # Turn on the eval mode
        self.Mymodel.eval()
        with torch.no_grad():
            for inputs, targets,emotion, ddp_features in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                emotions = emotion.to(self.args.device)
                # ddp_features = torch.tensor(ddp_features, dtype = torch.double).to(self.args.device)
                ddp_features = ddp_features.to(self.args.device)
                predicts = self.Mymodel(inputs,emotions, ddp_features)

                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                prds = torch.argmax(predicts, dim=1)
                n_correct += (prds == targets).sum().item()
                n_test += targets.size(0)
                # n_TP += (torch.argmax(predicts, dim=1) * targets == 1).sum().item()
                # x += (torch.argmax(predicts, dim=1) * torch.argmax(predicts, dim=1) == 0).sum().item()
                # x1 += (targets == 1).sum().item()
                # x0 += (torch.argmax(predicts, dim=1) == 1).sum().item()
                # n_TP_0 += (torch.argmax(predicts, dim=1) == targets).sum().item()
                # x1_0 += (targets == 0).sum().item()
                # x0_0 += (torch.argmax(predicts, dim=1) == 0).sum().item()
                # n_TP_2 += (torch.argmax(predicts, dim=1) * targets == 4).sum().item()
                # x1_2 += (targets == 2).sum().item()
                # x0_2 += (torch.argmax(predicts, dim=1) == 2).sum().item()
                prds_list.append(prds)
                labels_list.append(targets)

            prds_all = torch.cat(prds_list)
            labels_all = torch.cat(labels_list)
            f1 = f1_score(labels_all.cpu().detach().numpy(), prds_all.cpu().detach().numpy(), average = 'macro')
            precise = precision_score(labels_all.cpu().detach().numpy(), prds_all.cpu().detach().numpy(),
                                      average = 'macro')
            recall = recall_score(labels_all.cpu().detach().numpy(), prds_all.cpu().detach().numpy(),
                                  average = 'macro')
        # n_TP_0 = n_TP_0-n_TP_2-n_TP
        # pr = n_TP / x0
        # re = n_TP / x1
        # F1 = 2 * pr * re / (pr + re)
        # pr0 = n_TP_0 / x0_0
        # re0 = n_TP_0 / x1_0
        # F10 = 2 * pr0 * re0 / (pr0 + re0)
        # pr2 = n_TP_2 / x0_2
        # re2 = n_TP_2 / x1_2
        # F12 = 2 * pr2 * re2 / (pr2 + re2)
        return test_loss / n_test, n_correct / n_test, f1, precise, recall

    def _val(self, dataloader, criterion):
        n_classes=2
        target_num = torch.zeros((1, n_classes))  # n_classes为分类任务类别数量
        predict_num = torch.zeros((1, n_classes))
        acc_num = torch.zeros((1, n_classes))
        val_loss, n_correct, n_val, n_TP,n_FP,n_FN=0,0,0, 0, 0, 0,
        x,x1,x0=0,0,0
        # accuracy = evaluate.load("accuracy")
        # f1_metric = evaluate.load("f1")
        # recall = evaluate.load("recall")
        # precision = evaluate.load("precision")
        # Turn on the eval mode
        self.Mymodel.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                val_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_TP += (torch.argmax(predicts, dim=1)*targets == 1).sum().item()
                x+= (torch.argmax(predicts, dim=1)* torch.argmax(predicts, dim=1) ==0).sum().item()
                x1 += (targets == 1).sum().item()
                x0 += (torch.argmax(predicts, dim=1) == 1).sum().item()
                n_val += targets.size(0)

                _, predicted = predicts.max(1)
                pre_mask = torch.zeros(predicts.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
                tar_mask = torch.zeros(predicts.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)  # 得到数据中每类的数量
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量
                # F1 = f1_metric.compute(targets, predicts)
                # re = recall.compute(targets, predicts)
                # pr = precision.compute(targets, predicts)
        # re = acc_num / target_num
        # pr = acc_num / predict_num
        # F1 = 2 * re * pr / (re + pr)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
        # pr=n_TP/x0
        # re=n_TP/x1
        # F1=2*pr*re/(pr+re)

        return val_loss / n_val, n_correct / n_val


    def run(self):
        # Print the parameters of model
        # for name, layer in self.Mymodel.named_parameters(recurse=True):
        # print(name, layer.shape, sep=" ")
        train_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         val_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        tr_criterion = nn.HingeEmbeddingLoss()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(_params, lr = self.args.lr, momentum = 0.9)
        else:
            optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        l_tracc,l_teacc, l_valacc, l_trloss, l_teloss, l_epo,l_valloss  =[],[], [], [], [], [],[]
        l_tepr, l_tere, l_tef1,l_valpr,l_valre,l_valf1=[],[],[],[],[],[]
        best_loss, best_acc = 0, 0
        # Get the best_loss and the best_acc
        if self.args.wandb:
            config = dict(
                epochs = self.args.num_epoch,
                lr = self.args.lr,
                optimizer = self.args.optimizer,
                gram = self.args.gram
            )
            # wandb.init(
            #            project='{}_{}'.format(self.args.model_name,
            #                                      self.args.method_name,
            #                                      self.args.timestamp),
            #            name='{}_{}_{}'.format(self.args.model_name,
            #                                      self.args.method_name,
            #                                      self.args.timestamp))
            wandb.init(config = config,
                       project = '{}_{}'.format('roberta',
                                                'fnn'),
                       entity = 'xdu_ai')
            # wandb.define_metric("epoch")

        for epoch in range(self.args.num_epoch):
            train_loss, train_acc, train_f1, train_prec, train_recal = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, test_f1, test_prec, test_recal = self._test(test_dataloader, criterion)
            if self.args.wandb:
                wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
                           'train_prec': train_prec, 'train_reca': train_recal}, commit = False)
                wandb.log({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1,
                           'test_prec': test_prec, 'test_reca': test_recal}, commit = True)
            # val_loss , val_acc, valpr, valre, valf1 = self._val(val_dataloader, criterion)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                torch.save(self.Mymodel, "best-bert-bilstm.pth")
            l_epo.append(epoch), l_tracc.append(train_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
            # l_valloss.append(val_loss)
            # l_valpr.append(valpr), l_valre.append(valre), l_valf1.append(valf1)
            l_teacc.append(test_acc)
            # l_valacc.append(val_acc)
            # l_tepr.append(tepr), l_tere.append(tere), l_tef1.append(tef1)
            # if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            #     best_acc, best_loss = val_acc, val_loss
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            # self.logger.info('[val] loss: {:.4f}, acc: {:.2f}'.format(val_loss, val_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))

            # self.logger.info('[test] pr1: {:.2f}, re1: {:.2f}, f11: {:.2f}'.format(tepr, tere, tef1))
        wandb.finish()
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        # Draw the training process
        plt.plot(l_epo, l_tracc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('acc.png')

        # plt.plot(l_epo, l_valloss)
        # plt.ylabel('val_loss')
        # plt.xlabel('epoch')
        # plt.savefig('valloss.png')

        plt.plot(l_epo, l_teloss)
        plt.ylabel('test-loss')
        plt.xlabel('epoch')
        plt.savefig('teloss.png')
        plt.plot(l_epo, l_trloss)
        plt.ylabel('train-loss')
        plt.xlabel('epoch')
        plt.show()
        plt.savefig('trloss.png')
        data0={"l_tracc":l_tracc,"l_teacc":l_teacc,"l_trloss":l_trloss, "l_teloss":l_teloss, "l_epo":l_epo}
        print(data0)
        df = pd.DataFrame(data0)
        df.to_excel('128d-emoroberta-rnn-output.xlsx',index=False)


if __name__ == '__main__':
    # 这个是主要的代码文件
    logging.set_verbosity_error()
    args, logger = get_config()
    setup_seed(2024)
    # model_name = ['xlnet', 'bert', 'roberta', 'roberta', 'roberta', 'roberta']
    # method_name = ['none', 'none', 'fnn', 'textcnn', 'attention', 'bilstm']
    # model_name = ['bert', 'roberta', 'roberta', 'roberta', 'roberta']
    # method_name = ['none', 'fnn', 'textcnn', 'attention', 'bilstm']
    # model_name = ['roberta', 'roberta', 'roberta']
    # method_name = ['textcnn', 'attention', 'bilstm']
    # model_name = ['roberta', 'roberta', 'bert', 'roberta', 'roberta']
    # method_name = ['bilstm', 'fnn', 'none', 'attention', 'bilstm']
    # for mm, md in zip(model_name, method_name):
    #     args.model_name = mm
    #     args.method_name = md
    #     nb = Niubility(args, logger)
    #     nb.run()

    mm = 'roberta'
    md = 'fnn'
    args.model_name = mm
    args.method_name = md
    nb = Niubility(args, logger)
    nb.run()

    from tkinter import messagebox
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    window = tk.Toplevel(root)
    window.attributes('-topmost', True)
    messagebox.showinfo('提示', '程序运行完成!')
    window.destroy()
