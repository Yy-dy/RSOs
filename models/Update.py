#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class MyDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx]


class LocalUpdate(object):
    def __init__(self, args, dataset_x=None,dataset_y=None, idxs=None):
        self.args = args
        # self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        dataset = MyDataset(dataset_x[idxs], dataset_y[idxs])
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True) #dataset=[tensor1,tensor2,tensor3]

    def train(self, net):
        net.train()
        # train and update
        #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=self.args.momentum)
        #optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
        #optimizer = torch.optim.Adagrad(net.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

