import math

import numpy as np
from sklearn.svm import SVC
import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, base_model, num_classes, input_size, args):
        super().__init__()
        # 初始化函数，定义了模型的结构和参数
        # base_model: 底层的Transformer模型
        # num_classes: 分类任务的类别数
        # input_size: 输入特征的大小
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
            nn.Softmax(dim = 1)
        )

    def forward(self, inputs, *x):
        raw_outputs = self.base_model(**inputs).last_hidden_state[:, 0, :]
        predicts = self.fc(raw_outputs)

        return predicts


class Baseline_transformer(nn.Module):
    def __init__(self, base_model, num_classes, input_size, args):
        super().__init__()
        # 初始化函数，定义了模型的结构和参数
        # base_model: 底层的Transformer模型
        # num_classes: 分类任务的类别数
        # input_size: 输入特征的大小
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        # 线性层，用于将模型输出映射到类别空间
        # 输入大小为base_model.config.hidden_size + 1
        self.linear = nn.Linear(base_model.config.hidden_size + 1, num_classes)
        ddp_size = args.gram
        # lin_hidden_size = 256
        # self.linear = nn.Linear(base_model.config.hidden_size + 1 + ddp_size, num_classes)
        # self.linear_trans = nn.Linear(base_model.config.hidden_size+1 + ddp_size, lin_hidden_size)
        # self.fc = nn.Linear(lin_hidden_size, num_classes)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(0.5)
        # Softmax层，用于将线性输出转换为概率分布
        self.softmax = nn.Softmax()
        # 将base_model的参数设置为可训练
        for param in base_model.parameters():
            param.requires_grad = (True)

        # add by wyh
        # 创建一个 TransformerEncoderLayer 对象
        encoder_layer = nn.TransformerEncoderLayer(d_model = args.gram, nhead = 2, batch_first = True)

        # 创建一个 TransformerEncoder 对象，包含 2 个 encoder_layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)
        self.ddp_mlp = nn.Linear(args.gram, ddp_size)


    def forward(self, inputs, emotion, ddparser_feature):
        # 前向传播函数，定义了模型的计算过程
        # inputs: 输入数据
        # emotion: 情感信息
        # 获取底层Transformer模型的原始输出
        raw_outputs = self.base_model(**inputs)
        # The pooler_output is made of CLS --> FNN --> Tanh
        # The last_hidden_state[:,0] is made of original CLS
        # Method one
        # cls_feats  = raw_outputs.pooler_output
        # Method two
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        # emotion 张量的第一个维度（维度索引为 1）上增加一个维度。(batch_size, features) => (batch_size, 1, features)
        emotion = emotion.unsqueeze(1)
        emotion = emotion.float()

        ddparser_feature1 = torch.mean(self.transformer_encoder(ddparser_feature), dim = 1)
        # 可以选择取平均或展平
        # ddparser_feature1 = torch.flatten(ddparser_feature, start_dim = 1, end_dim = 2)
        # 可在平均之后在接一个 mlp
        # ddparser_feature2 = self.ddp_mlp(ddparser_feature1)
        ddparser_feature2 = ddparser_feature1

        emovec = torch.cat([emotion, cls_feats, ddparser_feature2], axis=1)
        last_feature = self.linear_trans(self.dropout(emovec))
        predicts = self.softmax(self.fc(last_feature))
        # predicts = self.softmax(self.linear(self.dropout(emovec)))
        return predicts


# Bert + FNN
class Transformer(nn.Module):
    def __init__(self, base_model, num_classes, input_size, args):
        super().__init__()
        # 初始化函数，定义了模型的结构和参数
        # base_model: 底层的Transformer模型
        # num_classes: 分类任务的类别数
        # input_size: 输入特征的大小
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        # 线性层，用于将模型输出映射到类别空间
        # 输入大小为base_model.config.hidden_size + 1
        ddp_size = args.gram
        lin_hidden_size = 256
        self.linear = nn.Linear(base_model.config.hidden_size + 1 + ddp_size, num_classes)
        self.linear_trans = nn.Linear(base_model.config.hidden_size + 1 + ddp_size, lin_hidden_size)
        # 去除情感特征
        # self.linear = nn.Linear(base_model.config.hidden_size + ddp_size, num_classes)
        # self.linear_trans = nn.Linear(base_model.config.hidden_size + ddp_size, lin_hidden_size)
        # 去除句法依存
        # self.linear = nn.Linear(base_model.config.hidden_size + 1, num_classes)
        # self.linear_trans = nn.Linear(base_model.config.hidden_size + 1, lin_hidden_size)

        self.fc = nn.Linear(lin_hidden_size, num_classes)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(0.5)
        # Softmax层，用于将线性输出转换为概率分布
        self.softmax = nn.Softmax()
        # 将base_model的参数设置为可训练
        for param in base_model.parameters():
            param.requires_grad = (True)

        # add by wyh
        # 创建一个 TransformerEncoderLayer 对象
        encoder_layer = nn.TransformerEncoderLayer(d_model = args.gram, nhead = 2, batch_first = True)

        # 创建一个 TransformerEncoder 对象，包含 2 个 encoder_layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)
        self.ddp_mlp = nn.Linear(args.gram, ddp_size)

        # 输出用 transformer
        size = 768
        encoder_fc = nn.TransformerEncoderLayer(d_model = size, nhead = 8, batch_first = True)
        self.transformer_fc = nn.TransformerEncoder(encoder_fc, num_layers = 2)
        self.block = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(base_model.config.hidden_size, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(16, num_classes),
            nn.Softmax(dim = 1)
        )
        self.transformer_output = nn.Sequential(
            nn.Linear(size, num_classes, bias = True),
            nn.Softmax(dim = 1)
        )

    # 使用 transformer 作为输出，修改版
    def forward(self, inputs, emotion, ddp_feature):
        # 前向传播函数，定义了模型的计算过程
        # inputs: 输入数据
        # emotion: 情感信息
        # 获取底层Transformer模型的原始输出
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        a = tokens.shape[0]
        emotion = emotion.unsqueeze(1)
        emotion = emotion.expand(a, 768)
        emotion = emotion.unsqueeze(1)
        # emotion = emotion.unsqueeze(1)
        emotion = emotion.float()

        ddp_feature = torch.mean(self.transformer_encoder(ddp_feature), dim = 1)
        ddp_feature = ddp_feature.unsqueeze(-1)
        ddp_feature = ddp_feature.expand(ddp_feature.shape[0], 10, 768)

        # tokens = torch.cat([tokens, emotion, ddp_feature], axis=1)
        # 去除句法特征
        # tokens = torch.cat([tokens, emotion], axis = 1)
        # 去除情感特征
        tokens = torch.cat([tokens, ddp_feature], axis = 1)


        # 取第一层
        # res = self.transformer_fc(self.dropout(tokens))[:, 0, :]
        # 取平均
        res = torch.mean(self.transformer_fc(self.dropout(tokens)), dim = 1)
        # predicts = self.block(res)
        predicts = self.transformer_output(res)

        # predicts = self.softmax(self.transformer_output(self.dropout(res)))

        return predicts
    # def forward(self, inputs, emotion, ddparser_feature):
    #     # 前向传播函数，定义了模型的计算过程
    #     # inputs: 输入数据
    #     # emotion: 情感信息
    #     # 获取底层Transformer模型的原始输出
    #     raw_outputs = self.base_model(**inputs)
    #     # The pooler_output is made of CLS --> FNN --> Tanh
    #     # The last_hidden_state[:,0] is made of original CLS
    #     # Method one
    #     # cls_feats  = raw_outputs.pooler_output
    #     # Method two
    #     cls_feats = raw_outputs.last_hidden_state[:, 0, :]
    #     # emotion 张量的第一个维度（维度索引为 1）上增加一个维度。(batch_size, features) => (batch_size, 1, features)
    #     emotion = emotion.unsqueeze(1)
    #     emotion = emotion.float()
    #
    #     ddparser_feature1 = torch.mean(self.transformer_encoder(ddparser_feature), dim = 1)
    #     # 可以选择取平均或展平
    #     # ddparser_feature1 = torch.flatten(ddparser_feature, start_dim = 1, end_dim = 2)
    #     # 可在平均之后在接一个 mlp
    #     # ddparser_feature2 = self.ddp_mlp(ddparser_feature1)
    #     ddparser_feature2 = ddparser_feature1
    #
    #     # 加入情感特征, 句法依存特征
    #     emovec = torch.cat([emotion, cls_feats, ddparser_feature2], axis=1)
    #     # 去除情感特征
    #     # emovec = torch.cat([cls_feats, ddparser_feature2], axis=1)
    #     # 移除句法依存特征
    #     # emovec = torch.cat([emotion, cls_feats], axis = 1)
    #
    #     # mlp
    #     # last_feature = (self.linear_trans(self.dropout(emovec))).relu()
    #     # predicts = self.softmax(self.fc(last_feature))
    #     # 下面这个暂时废弃
    #     # predicts = self.softmax(self.linear(self.dropout(emovec)))
    #
    #     # 使用 transformer 作为输出
    #     # 句法特征
    #     pad = torch.zeros((len(ddparser_feature2), (cls_feats.shape[1] - ddparser_feature2.shape[1]))).to('cuda')
    #     ddparser_feature2 = torch.concat([ddparser_feature2, pad], dim = 1)
    #     # 情感特征
    #     pad = torch.zeros((len(emotion), (cls_feats.shape[1] - emotion.shape[1]))).to('cuda')
    #     emotion = torch.concat([emotion, pad], dim = 1)
    #
    #     ddparser_feature2 = ddparser_feature2.unsqueeze(1)
    #     emotion = emotion.unsqueeze(1)
    #     cls_feats = cls_feats.unsqueeze(1)
    #     joint_res = torch.concat([cls_feats, emotion, ddparser_feature2], dim = 1)
    #     res = self.transformer_fc(self.dropout(joint_res))[:, 0, :]
    #     predicts = self.softmax(self.transformer_output(self.dropout(res)))
    #
    #     return predicts


class Gru_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Gru = nn.GRU(input_size=self.input_size,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        gru_output, _ = self.Gru(tokens)
        outputs = gru_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


# 带了emotion
# Try to use the softmax、relu、tanh and logistic
class Lstm_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(64 , 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs,emotion):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        # raw_outputs1 = self.base_model(**emotion)
        # cls_feats1 = raw_outputs1.last_hidden_state
        a = cls_feats.shape[0]
        emotion = emotion.unsqueeze(1)
        emotion = emotion.expand(a, 768)
        emotion = emotion.unsqueeze(1)
        emotion = emotion.float()
        emovec = torch.cat([emotion, cls_feats], axis=1)
        lstm_output, _ = self.Lstm(emovec)
        outputs = lstm_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


# 带了emotion
class BiLstm_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        # Open the bidirectional
        self.BiLstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=512,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        # add by wyh
        # 创建一个 TransformerEncoderLayer 对象
        encoder_layer = nn.TransformerEncoderLayer(d_model = 10, nhead = 2, batch_first = True)

        # 创建一个 TransformerEncoder 对象，包含 2 个 encoder_layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)
        self.ddp_mlp = nn.Linear(10, 10)
        # self.fc = nn.Sequential(nn.Dropout(0.5),
        #                         nn.Linear(512*2, self.num_classes),
        #                         nn.Softmax(dim=1))
        # self.block = nn.Sequential(
        #     # nn.Dropout(0.5),
        #     nn.Linear(512 * 2, 256),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(64, 16),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(16, num_classes),
        #     nn.Softmax(dim = 1)
        # )
        self.block = nn.Sequential(
            nn.Linear(512 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, num_classes),
            nn.Softmax(dim = 1)
        )
        for param in base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, emotion, ddp_feature):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        # raw_outputs1 = self.base_model(**emotion)
        # cls_feats1 = raw_outputs1.last_hidden_state
        a = cls_feats.shape[0]
        emotion = emotion.unsqueeze(1)
        emotion = emotion.expand(a, 768)
        emotion = emotion.unsqueeze(1)
        emotion = emotion.float()

        ddp_feature = torch.mean(self.transformer_encoder(ddp_feature), dim = 1)
        ddp_feature = ddp_feature.unsqueeze(-1)
        ddp_feature = ddp_feature.expand(ddp_feature.shape[0], 10, 768)

        emovec = torch.cat([cls_feats, emotion, ddp_feature], axis=1)
        outputs, _ = self.BiLstm(emovec)
        outputs = outputs[:, -1, :]
        outputs = self.block(outputs)

        return outputs

# 带了emotion
class Rnn_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=64,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(64, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs,emotion):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        a = cls_feats.shape[0]
        emotion = emotion.unsqueeze(1)
        emotion = emotion.expand(a, 768)
        emotion = emotion.unsqueeze(1)
        emotion = emotion.float()
        emovec = torch.cat([emotion, cls_feats], axis=1)
        outputs, _ = self.Rnn(emovec)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        outputs
        return outputs


# 带了emotion
class TextCNN_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.encode_layer = 12

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )
        self.block = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes),
            nn.Softmax(dim=1)
        )

        # add by wyh
        # 创建一个 TransformerEncoderLayer 对象
        gram = 10
        encoder_layer = nn.TransformerEncoderLayer(d_model = gram, nhead = 2, batch_first = True)

        # 创建一个 TransformerEncoder 对象，包含 2 个 encoder_layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)
        self.ddp_mlp = nn.Linear(gram, gram)

    def conv_pool(self, tokens, conv):
        tokens = conv(tokens)
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)
        tokens = F.max_pool1d(tokens, tokens.size(2))
        out = tokens.squeeze(2)
        return out

    def forward(self, inputs, emotion, ddp_feature):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)
        a = tokens.shape[0]
        emotion = emotion.unsqueeze(1)
        emotion = emotion.expand(a, 768)
        emotion = emotion.unsqueeze(1)
        emotion = emotion.unsqueeze(1)
        emotion = emotion.float()

        ddp_feature = torch.mean(self.transformer_encoder(ddp_feature), dim = 1)
        ddp_feature = ddp_feature.unsqueeze(-1)
        ddp_feature = ddp_feature.expand(ddp_feature.shape[0], 10, 768)
        ddp_feature = ddp_feature.unsqueeze(1)

        # 只含情感特征
        # emovec = torch.cat([emotion, tokens], axis=2)
        # 加入句法特征
        emovec = torch.cat([tokens, emotion, ddp_feature], axis = 2)
        out = torch.cat([self.conv_pool(emovec, conv) for conv in self.convs],
                        1)
        predicts = self.block(out)
        return predicts



class Transformer_CNN_RNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cnn_tokens = raw_outputs.last_hidden_state.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]
        rnn_tokens = raw_outputs.last_hidden_state
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts


# 带了emotion
class Transformer_Attention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        # self.block = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(768, 128),
        #     nn.Linear(128, 16),
        #     nn.Linear(16, num_classes),
        #     nn.Softmax(dim=1)
        # )
        self.block = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(base_model.config.hidden_size, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(16, num_classes),
            nn.Softmax(dim = 1)
        )
        # add by wyh
        # 创建一个 TransformerEncoderLayer 对象
        gram = 10
        encoder_layer = nn.TransformerEncoderLayer(d_model = gram, nhead = 2, batch_first = True)

        # 创建一个 TransformerEncoder 对象，包含 2 个 encoder_layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 2)
        self.ddp_mlp = nn.Linear(gram, gram)

    def forward(self, inputs, emotion, ddp_feature):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        a = tokens.shape[0]
        emotion = emotion.unsqueeze(1)
        emotion = emotion.expand(a, 768)
        emotion = emotion.unsqueeze(1)
        # emotion = emotion.unsqueeze(1)
        emotion = emotion.float()

        ddp_feature = torch.mean(self.transformer_encoder(ddp_feature), dim = 1)
        ddp_feature = ddp_feature.unsqueeze(-1)
        ddp_feature = ddp_feature.expand(ddp_feature.shape[0], 10, 768)

        tokens = torch.cat([emotion, tokens, ddp_feature], axis=1)
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)

        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)
        attention_output = torch.mean(attention_output, dim=1)

        predicts = self.block(attention_output)
        return predicts

class Transformer_CNN_RNN_Attention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)
        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        # Self-Attention
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)

        # TextCNN
        cnn_tokens = attention_output.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]

        rnn_tokens = tokens
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts

class biLstm_attention_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)
        # Open the bidirectional

        self.BiLstm = nn.LSTM(input_size=768,
                              hidden_size=320,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),

                                nn.Linear(320*2, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)



    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)
        attention_output = torch.mean(attention_output, dim=1)
        outputs, _ = self.BiLstm(attention_output)

        outputs = self.fc(outputs)
        return outputs


class BiLstm_attention_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Open the bidirectional

        self.BiLstm = nn.LSTM(input_size=768,
                              hidden_size=64,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(64 * 2, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        self.tanh1=nn.Tanh()
        self.w = nn.Parameter(torch.zeros(64 * 2))
        self.tanh2 = nn.Tanh()

        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        H, _ = self.BiLstm(cls_feats)

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)

        out = self.fc(out)  # [128, 64]
        return out