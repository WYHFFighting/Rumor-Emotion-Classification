# -*- coding: utf-8 -*-
import os
import numpy as np
from ddparser import DDParser
import jieba
import time
from gensim.models import KeyedVectors
from tqdm import tqdm
from anytree import Node, RenderTree
from functools import reduce
"""
environment: syntactic_analysis
"""


t1 = time.time()
ddp = DDParser()
jieba.load_userdict("user_dict.txt")


class Tree:
    def __init__(self, root=None):
        self.root = root


class Model:
    """
    1、只针对句子？。；！若句子内部有这些标点符号直接不比较，句子末尾标点符号去除
    2、对句子进行分词，依存句法分析，构建树时不加入标点符号部分
    3、先做词相似度
    4、比较两颗树：（1）去除标点符号
                （1）去除包括标点符号在内的停用词（修建树），标点符号
                （2）把ATT和名词语义综合，这个可以看做是分词失误。
    分析：
    1、SBV主谓关系
    2、VOB动宾关系
    3、POB介宾关系
    4、ADV状中关系
    5、CMP动补关系
    6、ATT定中关系
    7、F方位关系
    8、COO并列关系
    9、DBL兼语关系，主谓短语做宾语
    10、DOB双宾语
    11、VV连谓关系
    12、IC子句结构
    13、MT虚词
    14、HED核心
    """
    def __init__(self):
        self.sentence_delimiters = ["。", "！", "？", "；", ".", "!", "?", ";"]
        self.punctuations = ["，", "、", ",", "“", "”", "《", "》", "（", "）"]
        self.punctuations.extend(self.sentence_delimiters)
        self.stop_words = [line.strip() for line in open("中文停用词表.txt", "r", encoding="utf-8")]
        self.result = []
        print("word vectors loading!!!")
        self.t3 = time.time()
        self.vectors = KeyedVectors.load_word2vec_format(
            "tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt", binary=False)
        self.t4 = time.time()
        print(f"加载词向量用时： {self.t4 - self.t3}")
        print("word vectors loaded!!!")
        # print(self.stop_words)
        self.picture_counter = 1

    @staticmethod
    def sentence_parser(sentence):
        return ddp.parse_seg([jieba.lcut(sentence)])[0]

    def sentence_pre_process(self, sentence):
        """
        :param sentence:
        :return: 如果句子内部有句子分隔符，返回None；如果末尾有分割符，返回去除后的结果；否则直接返回句子
        """
        for delimiter in self.sentence_delimiters:
            if delimiter in sentence[:-1]:
                return None
        if sentence[-1] in self.sentence_delimiters:
            return sentence[:-1]
        return sentence

    @staticmethod
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

    def score_by_tree(self, root_1, root_2):
        """
        :param root_1:
        :param root_2:
        :return: 对于根节点的子节点，若是叶子节点，直接记录，若不是，往下遍历
        """
        children_1, children_2 = self.children_process(list(root_1.children)), self.children_process(list(root_2.children))

        if [a[0].split(" ")[0] for a in children_1] == [b[0].split(" ")[0] for b in children_2]:
            score = self.get_hed_similarity(root_1.name.split(" ")[1], root_2.name.split(" ")[1])
            if score < 0:
                return -1
            for index in range(len(children_1)):
                child_1_words = [c.split(" ")[1] for c in children_1[index]]
                child_2_words = [d.split(" ")[1] for d in children_2[index]]
                score *= self.get_words_similarity(child_1_words, child_2_words)
                if score < 0:
                    return -1
            print(children_1)
            print(children_2)
            return score
        return -1

    @staticmethod
    def add(x, y):
        return x + y

    def get_words_similarity(self, words_1, words_2):
        try:
            v1 = reduce(self.add, [self.vectors[word] for word in words_1]) / len(words_1)
            v2 = reduce(self.add, [self.vectors[word] for word in words_2]) / len(words_2)
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except KeyError:
            return -1

    def children_process(self, children):
        processed_children = []
        for child in children:
            child_res = self.child_process(child)
            if child_res:
                processed_children.append(child_res)
        return processed_children

    def child_process(self, child):
        res = []
        if child.name.startswith("ADV") or child.name.startswith("ATT") or child.name.startswith("SBV") or child.name.startswith("VOB"):
            res.append(child.name)
        if not child.is_leaf:
            for grandchild in child.children:
                grandchild_res = self.child_process(grandchild)
                if grandchild_res:
                    res.extend(grandchild_res)
        return res

    def get_hed_similarity(self, word_1, word_2):
        try:
            return np.dot(self.vectors[word_1], self.vectors[word_2]) / (np.linalg.norm(self.vectors[word_1]) * np.linalg.norm(self.vectors[word_2]))
        except KeyError:
            return -1

    def make_hed_score(self, s1_info, s2_info):
        return self.get_words_similarity(s1_info["word"][s1_info["head"].index(0)], s2_info["word"][s2_info["head"].index(0)])

    def make_specific_position_score(self, s1_info, s2_info, position):
        return self.get_words_similarity(s1_info["word"][s1_info["deprel"].index(position)], s2_info["word"][s2_info["deprel"].index(position)])

    def make_score(self, s1_info, s2_info):
        """
        只处理句子中有一个SBV和VOB的句子，不符合此规范以及句子中词向量不可查的返回-1
        :param s1_info:
        :param s2_info:
        :return:
        """
        if s1_info["deprel"].count("SBV") != 1 or s1_info["deprel"].count("VOB") != 1 or s2_info["deprel"].count("SBV") != 1 or s2_info["deprel"].count("VOB") != 1:
            return -1
        hed_score = self.make_hed_score(s1_info, s2_info)
        sbv_score = self.make_specific_position_score(s1_info, s2_info, "SBV")
        vob_score = self.make_specific_position_score(s1_info, s2_info, "VOB")
        if hed_score == -1 or sbv_score == -1 or vob_score == -1:
            return -1
        return [round(hed_score, 2), round(sbv_score, 2), round(vob_score, 2), round(hed_score * sbv_score * vob_score, 2)]

    def predict(self, sentences_pair):
        sentence_1 = self.sentence_pre_process(sentences_pair[0])
        sentence_2 = self.sentence_pre_process(sentences_pair[1])
        if not sentence_1 or not sentence_2:
            return -1
        label = int(sentences_pair[2])
        s1_info = self.sentence_parser(sentence_1)
        s2_info = self.sentence_parser(sentence_2)
        score = self.score_by_tree(self.make_tree(s1_info, print_structure=False), self.make_tree(s2_info, print_structure=False))

        print(s1_info)
        print(s2_info)
        print(label)
        print(score)
        if score != -1:
            # print(s1_info)
            # print(s2_info)
            # print(label)
            # print(score)
            self.result.append({
                "sentence_1": sentence_1,
                "sentence_2": sentence_2,
                "s1_info": s1_info,
                "s2_info": s2_info,
                "label": label,
                "score": score
            })

    def print_structure(self, sentence):
        print(sentence)
        info = self.sentence_parser(sentence)
        print(info)
        self.make_tree(info)
        print()

    def predict_test_dataset(self, dataset_file_name="dev.txt"):
        dataset_file_name = os.path.join(os.getcwd(), "测试用例", dataset_file_name)
        with open(dataset_file_name, "r", encoding="utf-8") as f:
            for index, line in tqdm(enumerate(f)):
                # print(index)
                self.predict(line.replace("\n", "").split("\t"))
                # sentence_1, sentence_2, _ = line.split("\t")
                # self.print_structure(sentence_1)
                # self.print_structure(sentence_2)

    def evaluate(self):
        threshold_list = [round((0.3 + i * 0.02), 2) for i in range(35)]
        for threshold in threshold_list:
            correct = 0
            size = len(self.result)
            for unit in self.result:
                if (unit["label"] == 1) and (unit["score"] >= threshold):
                    correct += 1
                elif (unit["label"] == 0) and (unit["score"] < threshold):
                    correct += 1
            accuracy = round(correct / size, 4)
            print(f"correct: {correct}, size: {size}, accuracy: {accuracy}, threshold: {threshold}")

    def peek(self):
        sentences = ["世界上最高的人是谁", "京东上运动服的价格是多少", "中国里最高的人是谁", "中国外最高的人是谁", "中国最高的人是谁"]
        for sentence in sentences:
            print(self.sentence_parser(sentence))


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')

    model = Model()
    text1 = "我今天上学迟到了"
    text2 = "我昨天来教室晚了一会儿"
    # model.predict([text1, text2, 0])

    print('------------------')
    text1 = "我今天上学迟到了"
    text2 = "我今天来教室晚了一会儿"
    # model.predict([text1, text2, 1])

    r1 = model.sentence_parser("我今天上学迟到了")
    print(r1)
    model.make_tree(r1, print_structure = True)
    r2 = model.sentence_parser("我今天来教室晚了一会儿")
    print(r2)
    model.make_tree(r2, print_structure = True)

# print(model.sentence_parser("导师今天来我家了，我感到非常荣幸"))
# model.make_tree(model.sentence_parser("导师今天来我家了，我感到非常荣幸"), print_structure=True)

# m.make_tree(m.sentence_parser("电脑微信和手机微信可以同步吗"), print_structure=True)
# m.predict_test_dataset()
# m.evaluate()
# m.predict_test_dataset()

# inf = m.sentence_parser("河南农业大学艺术中心的艺术培训都有那些课程？")
# r = m.make_tree(inf)
# print(list(r.children))
# print(r.is_leaf)
# m.score_by_tree(m.make_tree(m.sentence_parser("河南农业大学艺术中心的艺术培训都有那些课程？")), m.make_tree(m.sentence_parser("我是叔叔的什么人")))
# m.score_by_tree(m.make_tree(m.sentence_parser("杨幂现在在拍什么戏？")), m.make_tree(m.sentence_parser("杨幂现在在做什么？")))
# m.score_by_tree(m.make_tree(m.sentence_parser("韭菜多吃什么好处")), m.make_tree(m.sentence_parser("多吃韭菜有什么好处")))
# m.score_by_tree(m.make_tree(m.sentence_parser("叔叔是什么人")), m.make_tree(m.sentence_parser("我是叔叔的什么人")))
# m.make_tree(m.sentence_parser("杨幂现在在拍什么戏？"))
# m.make_tree(m.sentence_parser("杨幂现在在做什么？"))
# m.make_tree(m.sentence_parser("韩语的加油怎么写？"))
# m.make_tree(m.sentence_parser("韩语加油怎么说"))
# m.make_tree(m.sentence_parser(""))

# info_global = m.sentence_parser("京东上运动服的价格是多少")
# print(info_global)
# m.make_tree(info_global)
# # print(m.sentence_pre_process("中国和日本是邻国,这你都不知道?"))
# example = [""]
# # print(m.predict(["中国的邻国是日本。对吧", "中国的邻国是日本。", 1]))
# m.predict_test_dataset()
# m.evaluate()
# print(["a", "b", "a", "b", "d", "d", "e"].count("f"))
# print(m.get_words_similarity("省", "市"))
# print(m.get_words_similarity("省", "村"))
# print(m.get_words_similarity("省", "县"))
# print(m.get_words_similarity("市", "县"))
# print(m.get_words_similarity("国", "省"))
print()

t2 = time.time()
print(f"程序运行时间: {t2 - t1}")

