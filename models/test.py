#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import MyDataset


def test_img(net_g, dataset_x, dataset_y, args):
    net_g.eval()
    # 测试
    test_loss = []
    test_pm = []


    dataset = MyDataset(dataset_x[0], dataset_y[0])  # 只选择了其中一个卫星数据






    data_loader = DataLoader(dataset, batch_size=args.bs)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()

        # 获取模型输出
        predictions = net_g(data)

        # 累加每个batch的损失（使用MSE作为回归任务的损失）
        batch_loss = F.mse_loss(predictions, target, reduction='sum').item()
        test_loss += batch_loss

    # 计算平均损失
    test_loss /= len(data_loader.dataset)

    # 打印测试信息
    if args.verbose:
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return test_loss

