#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

# 定义模型-输入特征为14
class LSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=16, num_layers=8):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([nn.LSTM(input_size, hidden_size, batch_first=True)] +
                                    [nn.LSTM(hidden_size, hidden_size, batch_first=True) for _ in
                                     range(num_layers - 1)])
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out = x.unsqueeze(1)  # 将输入的维度从 (batch_size, 14) 转换为 (batch_size, 1, 14)
        for layer in self.layers:
            out, (h0, c0) = layer(out, (h0[:layer.num_layers], c0[:layer.num_layers]))  # 动态调整隐藏状态的维度

        out = self.fc(out[:, -1, :])

        return out
class RNN(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, output_size=1, seq_len=1000):
        """
        结合 RNN 和 MLP 结构的回归模型
        :param input_size: 输入特征维度
        :param hidden_size: RNN 隐藏层维度
        :param num_layers: RNN 层数
        :param output_size: 输出特征维度
        :param seq_len: 序列长度
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # 定义 RNN 层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='relu',
            batch_first=True
        )

        
        self.fc= nn.Linear(hidden_size,  output_size)  # 最终输出

    def forward(self, x, h_state=None):
        """
        前向传播
        :param x: 输入数据，形状为 (batch_size, seq_len, input_size)
        :param h_state: 初始隐藏状态
        :return: 输出和下一个隐藏状态
        """
        if h_state is None:
            h_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.unsqueeze(1) 
        # RNN 前向传播
        #print(x.shape)
        rnn_out, h_state_next = self.rnn(x, h_state)

        # 取 RNN 最后一个时间步的输出（适用于回归问题）
        rnn_out_last = rnn_out[:, -1, :]  # (batch_size, hidden_size)

        # 通过 MLP 进行特征变换
        output = self.fc(rnn_out_last)

        return output

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # 输入特征为14，输出为1的回归问题
        self.fc1 = nn.Linear(14, 64)  # 输入层到隐藏层1
        self.relu1 = nn.ReLU()        # 激活函数1
        self.fc2 = nn.Linear(64, 128) # 隐藏层1到隐藏层2
        self.relu2 = nn.ReLU()        # 激活函数2
        self.fc3 = nn.Linear(128, 64) # 隐藏层2到隐藏层3
        self.relu3 = nn.ReLU()        # 激活函数3
        self.fc4 = nn.Linear(64, 1)   # 输出层
        
    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

class MLPWX(nn.Module):
    def __init__(self, dim_in=14, dim_hidden=64, dim_out=1, dropout_prob=0.5):
        super(MLPWX, self).__init__()  # 正确使用 super()
        # 输入层：dim_in -> dim_hidden
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        
        # 可选的：BatchNorm 层（对隐藏层进行批归一化）
        #self.batch_norm = nn.BatchNorm1d(dim_hidden)
        
        # Dropout 层：用于防止过拟合
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # 隐藏层：dim_hidden -> dim_out（回归问题，输出1个值）
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        # 输入数据没有图像的维度，因此无需展平操作
        # 直接通过输入层和后续层进行处理
        
        # 输入层到隐藏层
        x = self.layer_input(x)
        
        # BatchNorm 层（归一化）
        #x = self.batch_norm(x)
        
        # Dropout 层
        x = self.dropout(x)
        
        # 隐藏层到输出层
        x = self.layer_hidden(x)
        
        # 输出层的输出是一个连续值（回归任务）
        
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128, 128)  # 修改为相同的输入和输出维度
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        #print(f"Input shape before unsqueeze: {x.shape}")  
        x = x.unsqueeze(1)  # 变为 (batch_size, 1, 14)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    print("cnn成功")

class CNNMnist(nn.Module):
    
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
class CNNWX(nn.Module):
    def __init__(self):
        super(CNNWX, self).__init__()
        # 修改：输入特征数量为14，输出层为1
        self.fc_layers = nn.ModuleList([
            nn.Linear(14 if i == 0 else 64, 64) for i in range(8)
        ])
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        # 修改：输入数据reshape成(batch_size, 14)，以适应新的输入特征数量
        x = x.view(-1, 14)  # 输入特征数量改为14
        for fc_layer in self.fc_layers:
            x = torch.relu(fc_layer(x))  # 每一层都经过ReLU激活
        x = self.fc_out(x)  # 输出层
        return x'''


# 定义 GRU 模型类
class GRU(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, output_size=1, seq_len=100, dropout_rate=0.2):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # 定义 GRU 层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state=None):
        x = x.unsqueeze(1)
        if h_state is None:
            h_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU 前向传播
        gru_out, h_state_next = self.gru(x, h_state)

        # 取 GRU 最后一个时间步的输出（适用于回归问题）
        gru_out_last = gru_out[:, -1, :]  # (batch_size, hidden_size)

        # 应用 Dropout
        gru_out_last = self.dropout(gru_out_last)

        output = self.fc(gru_out_last)

        return output

class Transformer(nn.Module):
    def __init__(self, input_size=14, d_model=32, num_heads=4, num_layers=3, hidden_dim=64, output_size=1):
        super(Transformer, self).__init__()

        self.input_fc = nn.Linear(input_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.input_fc(x)  # 线性映射到 d_model 维度
        x = x.permute(1, 0, 2)  # 变换为 (seq_length, batch_size, d_model)

        x = self.transformer_encoder(x)  # 通过 Transformer 编码器
        x = x.mean(dim=0)  # 对序列维度求均值 (类似 Global Average Pooling)

        output = self.fc_out(x)  # 输出预测值
        return output

class CNNWX(nn.Module):
    def __init__(self):
        super(CNNWX, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128, 128)  # 修改为相同的输入和输出维度
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 将输入的维度从 (batch_size, 14) 转换为 (batch_size, 1, 14)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x