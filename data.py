import re
from functools import partial
import jieba
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from ddparser import DDParser
from anytree import Node, RenderTree
import numpy as np
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
# from gensim.models import KeyedVectors


# Make MyDataset
class MyDataset(Dataset):
    def __init__(self, sentences, labels, method_name, model_name, result):
        self.sentences = sentences
        self.labels = labels
        self.method_name = method_name
        self.model_name = model_name
        self.result = result
        dataset = list()
        index = 0
        # jieba.load_userdict(r"dictionary.txt")
        for data in sentences:
            a = result.loc[result['文本'] == data, 'q']
            emotion = a.iloc[0]
            # 增加去除符号、非法字符
            # data = re.sub(r'[\u4e00-\u9fa50-9]+', '', data)
            data = ''.join(re.findall(r'[\u4e00-\u9fa50-9]+', data, re.S))

            tokens = jieba.lcut(data)
            labels_id = labels[index]
            index += 1
            dataset.append((tokens, labels_id, emotion))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)


# Make tokens for every batch
def my_collate(batch, tokenizer, ddp, vectorizer = None):
    tokens, label_ids, emotion = map(list, zip(*batch))

    # if tokenizer:
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=128,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    embedd_list = np.array(text_ids['input_ids'])
    # else:
    #     embedd_list = []
    #     for line in tokens:
    #         tt = [vectorizer.word_vec(w) for w in line]
    #         embedd_list.append(tt)
        # embedd_list = np.array([vectorizer.word_vec(w) for w in tokens])

    corrected_embedd_list = []
    tokens = []
    def check_bert(ss):
        if ss == '[CLS]' or ss == '[SEP]' or ss == '[PAD]':
            return False
        return True
    for cnt, item in enumerate(text_ids.encodings):
        temp_embed = [x.item() for cc, x in enumerate(embedd_list[cnt]) if item.sequence_ids[cc] == 0]
        corrected_embedd_list.append(temp_embed)

        words = [x for x in item.words if x is not None]
        detached_tokens = [x for x in item.tokens if check_bert(x)]
        tokens_num = np.unique(words)
        start = 0
        second_temp = []
        for i in sorted(tokens_num):
            temp = ''
            while words[start] == i:
                temp += detached_tokens[start]  # [0 for x in text_ids.encodings[20].sequence_ids if x == 0]
                start += 1
                if start >= len(detached_tokens):
                    break
            if len(second_temp) == 0:
                second_temp = [temp]
            else:
                second_temp.append(temp)
        if len(tokens) == 0:
            tokens = [second_temp]
        else:
            tokens.append(second_temp)


    # tokens 得到的是分词后的结果

    ddparser_feature = np.array([])

    for embedd, item in zip(corrected_embedd_list, tokens):
        res = ddp.parse_seg([item])[0]
        res = tree2embedd(embedd, res, vectorizer).reshape(1, -1, 10)
        if len(ddparser_feature) == 0:
            ddparser_feature = res
        else:
            ddparser_feature = np.concatenate([ddparser_feature, res], axis = 0)
                    # temp = np.concatenate([temp, word2embed[item]])
    # for item in batch:

    # emotion = [round(x) for x in emotion]
    # emotion = [str(x) for x in emotion]
    # emotion = [[x] for x in emotion]
    # emotion_ids = tokenizer(emotion,
    #                      padding=True,
    #                      truncation=True,
    #                      max_length=10,
    #                      is_split_into_words=True,
    #                      add_special_tokens=True,
    #                      return_tensors='pt')
    # print(1,text_ids['position_ids'])
    # print(2,text_ids['attention_mask'])
    # print(3,text_ids['input_ids'])
    return text_ids, torch.tensor(label_ids), torch.tensor(emotion), torch.tensor(ddparser_feature, dtype = torch.float)


def tree2embedd(embedd, res, vectorizer):
    word = res['word']
    head = res['head']

    head2pos = {v: k for k, v in enumerate(sorted(np.unique(head)))}
    # print(len(np.unique(head)))
    # print(word, '**********************************')
    # word2head = {}
    # for k, v in zip(word, head):
    #     print(k, v)
    # word2head = {k: v for k, v in zip(word, head)}
    # word2head = [(k, v) for k, v in zip(word, head)]
    keys = []
    values = []
    for k, v in zip(word, head):
        keys.append(k)
        values.append(v)
    word2head = [keys, values]
    # word2head = {(k, v) for k, v in zip(word, head)}
    # print(word2head)
    # word2pos = {k: head2pos[word2head[k]] for k in word}
    # word2pos = [(k, head2pos[word2head[k]]) for k, v in zip(word, head)]
    keys = []
    values = []
    # for k, v in zip(word, head):
    #     keys.append(k)
    #     values.append(head2pos[word2head[k]])
    for i, w in enumerate(word):
        keys.append(w)
        values.append(head2pos[word2head[1][i]])
    word2pos = [keys, values]

    pos2word = {}
    # for k in word2pos.keys():
    #     if not pos2word.get(word2pos[k]):
    #         pos2word[word2pos[k]] = [k]
    #     else:
    #         pos2word[word2pos[k]].append(k)
    for i1, k in enumerate(word2pos[0]):
        if not pos2word.get(word2pos[1][i1]):
            pos2word[word2pos[1][i1]] = [k]
        else:
            pos2word[word2pos[1][i1]].append(k)

    if vectorizer:
        word2embed = {}
        for w in word:
            try:
                word2embed[w] = np.mean(vectorizer.word_vec(w))
            except:
                word2embed[w] = 0
    else:
        # 如果是去除分词特征 在这里进行处理
        word2embed = {}
        pos = 0
        # embedd = embedd[1: -1]
        pattren = re.compile(u'[\u4e00-\u9fa5]')
        for w in word:
            # w = '今天'
            if not re.search(pattren, w):
                L = 1
            # 处理 bert 无法编码的汉字
            elif '[' in w:
                L = len(w)
                regex_res = re.findall(r'(\[.*?\])', w)
                for t in regex_res:
                    L -= len(t) - 1
            else:
                L = len(w)
            for i in range(L):
                if i == 0:
                    word2embed[w] = embedd[pos]
                else:
                    word2embed[w] += embedd[pos]
                    # 如果去除分词特征, 这里不用累加, 直接拼接即可
                pos += 1
            # word2embed[w] = float(word2embed[w]) / L
            word2embed[w] = word2embed[w] // L

    for k in sorted(pos2word.keys()):
    # for k in pos2word[0]:
        if k == 0:
            res = np.zeros((1, 10))
            res[0][0] = np.array([word2embed[pos2word[k][0]]])
            # res[0][0] = np.array([word2embed[pos2word[k]]])
            # res = np.zeros(128)
            # res[0] = np.array([word2embed[pos2word[k][0]]])
        else:
            temp = np.zeros((1, 10))
            # temp = np.zeros(128)
            # item = '今', '天'
            for i, item in enumerate(pos2word[k]):
                if i >= 10:
                    break
                temp[0][i] = word2embed[item]
                # temp[i] = word2embed[item]
            res = np.concatenate([res, temp], axis = 0)
    std_lay = 64
    if res.shape[0] >= std_lay:
        return res[: std_lay, :]
    else:
        for i in range(std_lay - res.shape[0]):
            res = np.concatenate([res, np.zeros((1, 10))], axis = 0)
        return res
    # return res


# Load dataset
def load_dataset(tokenizer, train_batch_size, test_batch_size, val_batch_size, model_name, method_name, workers, test_shuffle = True, tok = True):
    data = pd.read_excel('谣言.xlsx')
    # 调试时使用
    # data = pd.read_excel('谣言.xlsx')[:100]
    data1 = data[['类别', '真相文本', '真相文本情感']]
    data1.rename(columns={'真相文本': '文本', '真相文本情感': '文本情感'}, inplace=True)
    data1['真相'] = 1
    data2 = data[['类别', '谣言文本', '谣言文本情感']]
    data2.rename(columns={'谣言文本': '文本', '谣言文本情感': '文本情感'}, inplace=True)
    data2['真相'] = 0
    result = pd.concat([data1, data2], axis=0)
    result.reset_index(drop=True, inplace=True)
    result['类别标签'] = 0
    result.loc[result['类别'] == '政治军事', '类别标签'] = 0
    result.loc[result['类别'] == '营养健康', '类别标签'] = 1
    result.loc[result['类别'] == '疾病防治', '类别标签'] = 2
    result.loc[result['类别'] == '社会生活', '类别标签'] = 3
    result.loc[result['类别'] == '科学技术', '类别标签'] = 4

    result['q'] = result['文本情感']+1
    len1 = int(len(list(result['类别'])))
    labels = list(result['真相'])[0:len1]
    sentences = list(result['文本'])[0:len1]
    emotionlabels = list(result['文本情感'])[0:len1]

    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=0.8, random_state=42, shuffle = test_shuffle)
    # tr_sen, te_val_sen, tr_lab, te_val_lab = train_test_split(sentences, labels, train_size=0.8, random_state=42)
    # te_sen, val_sen, te_lab, val_lab = train_test_split(te_val_sen, te_val_lab, train_size=0.5, random_state=42)
    # Dataset
    train_set = MyDataset(tr_sen, tr_lab, method_name, model_name,result)
    test_set = MyDataset(te_sen, te_lab, method_name, model_name,result)
    # val_set = MyDataset(val_sen, val_lab, method_name, model_name)
    # DataLoader
    ddp = DDParser()
    if not tok:
        vectorizer = KeyedVectors.load_word2vec_format(
            "test_vec.txt", binary = False)
        # vectorizer  = KeyedVectors.load_word2vec_format(
        #     "tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt", binary=False)
        collate_fn = partial(my_collate, tokenizer = tokenizer, ddp = ddp, vectorizer = vectorizer)
        train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle = test_shuffle,
                                  num_workers = workers,
                                  collate_fn = collate_fn, pin_memory = True)
        test_loader = DataLoader(test_set, batch_size = test_batch_size, shuffle = test_shuffle, num_workers = workers,
                                 collate_fn = collate_fn, pin_memory = True)
        # val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True, num_workers=workers,
        #                          collate_fn=collate_fn, pin_memory=True)
        return train_loader, test_loader

    collate_fn = partial(my_collate, tokenizer=tokenizer, ddp = ddp)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=test_shuffle, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=test_shuffle, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True, num_workers=workers,
    #                          collate_fn=collate_fn, pin_memory=True)
    return train_loader, test_loader


def make_tree(info, print_structure=True):
    word = info["word"]
    starts = info["head"]
    deprel = info["deprel"]
    root = Node(" ".join(["HED", word[deprel.index("HED")], str(deprel.index("HED") + 1)]))
    stack = [root]
    while stack:
        parent = stack[-1]
        stack.remove(parent)
        parent_pos = int(parent.name.split(" ")[-1])
        for index, start in enumerate(starts):
            if start == parent_pos:
                son_node = Node(" ".join([deprel[index], word[index], str(index + 1)]), parent=parent)
                stack.append(son_node)

    if print_structure:
        for pre, fill, node in RenderTree(root):
            print("%s%s" % (pre, node.name))

    return root


if __name__ == '__main__':
    from main2 import Niubility
    from config import get_config
    args, logger = get_config()
    nb = Niubility(args, logger)
    # nb.run()
    # nb = Niubility(args, logger)
    # train_dataloader, test_dataloader = load_dataset(tokenizer = nb.tokenizer, train_batch_size = nb.args.train_batch_size,
    #                                                  test_batch_size = nb.args.test_batch_size, val_batch_size = nb.args.test_batch_size,
    #                                                  model_name = nb.args.model_name, method_name = nb.args.method_name,
    #                                                  workers = nb.args.workers)
    train_dataloader, test_dataloader = load_dataset(tokenizer = nb.tokenizer,
                                                     train_batch_size = 64,
                                                     test_batch_size = 64,
                                                     val_batch_size = 64,
                                                     model_name = nb.args.model_name, method_name = nb.args.method_name,
                                                     workers = nb.args.workers, test_shuffle = False, tok = False)
    # for train in train_dataloader:

    next(iter(train_dataloader))
    # for item, t, e in train_dataloader:
    #     print(item[0]['input_ids'][0])

        # break
