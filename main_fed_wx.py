#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar,CNNWX,LSTM,GRU,RNN,Transformer
from models.Fed import FedAvg
from models.test import test_img
import torch.nn.functional as F

import os
import gzip
import pickle
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset

from models.Update import MyDataset

def validate(args, model,val_x, val_y, idx):
    model.eval()
    val_loss = 0
    true_y = []
    predicted_y = []
    time_list = []  # 用于存储时间数据
    criterion = F.mse_loss

    dataset = MyDataset(val_x[idx], val_y[idx])
    val_loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=True)

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            val_loss += criterion(output, target).item()
            # 假设data的第一个元素是时间
            time_list.extend(data[:, 0].cpu().numpy())  # 提取时间并存入列表

            true_y.extend(target.cpu().numpy())
            predicted_y.extend(output.cpu().numpy())

    torch.cuda.empty_cache()

    return val_loss, true_y, predicted_y, time_list  # 返回时间列表


def get_PM(y_pred, y_test, y_mean, y_std):
    # 将y转换为二维ndarray
    y_pred = np.vstack(y_pred)
    y_test = np.vstack(y_test)

    # 下面是计算LiBin论文的评价指标
    P_z = 0
    P_m = 0
    for i in range(y_test.shape[0]):
        P_z += np.square((y_test[i] - y_pred[i]) * y_std)
        P_m += np.square((y_test[i]) * y_std + y_mean)
    PM = 1 - np.sqrt(P_z / P_m)
    return PM



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'wx'
    args.dataset_path = '/root/autodl-tmp/.autodl/CNN+MLP+FL/federated-learning-master-李/data/wx/1217/'
    args.model = 'T'

    # 指定文件夹路径
    file_list = os.listdir(args.dataset_path) # 获取指定文件夹下所有文件列表
    wx_list = []
    for file_name in file_list:
        # 检查文件是否以".txt"为扩展名
        if file_name.endswith(".pkl"):  
            # 获取编号
            wx_list.append(file_name[:-4])
    log_path = './save/'
    wx_list = ['17795','17806','17717']#17795,18416,18427,18274,18443,16139,16142,16143,16182,18215
    for args.feature in [1]:
        logging.basicConfig(
            filename=log_path
                    + 'train_{}_{}_{}_{}_lr{}_m{}_fea{}_k{}_adam_dynamic_NODLF_{}.png'.
                    format(args.model, args.epochs,args.local_ep,args.local_bs,
                        args.lr,args.momentum, args.feature, args.num_users,wx_list[2])
                    +'.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s' #输出格式
        )
        logging.info('Processing wx_list: {}'.format(wx_list))
        logging.info('feature:{}'.format(args.feature))
        # 依次读取卫星数据
        dataset_train_x = []
        dataset_train_y =[]
        dataset_val_x = []
        dataset_val_y = []
        dataset_test_x = []
        dataset_test_y = []
        y_mean = []
        y_std = []
        print(args.feature)
        for wx_name in wx_list:
            print(wx_name)
            # load dataset and split users
            if args.dataset == 'wx':
                # 定义数据加载
                def load_raw_data(s_name):
                    dateset_name = args.dataset_path + s_name + '.pkl'
                    with open(dateset_name, 'rb') as f:
                        train_set, valid_set, test_set,(x_mean, x_std), (y_mean, y_std) = pickle.load(f)

                    def numpy_data(data_xy):
                        data_x, data_y = data_xy
                        data_x_numpy = np.asarray(data_x, dtype=np.float32)
                        data_y_numpy = np.asarray(data_y, dtype=np.float32)
                        return data_x_numpy, data_y_numpy

                    test_set_x, test_set_y = numpy_data(test_set)
                    valid_set_x, valid_set_y = numpy_data(valid_set)
                    train_set_x, train_set_y = numpy_data(train_set)

                    # 提取y的其中1列
                    train_set_y = train_set_y[:, [args.feature]]
                    valid_set_y = valid_set_y[:, [args.feature]]
                    test_set_y = test_set_y[:, [args.feature]]
                    y_mean, y_std = y_mean[args.feature], y_std[args.feature]

                    rval = [(torch.from_numpy(train_set_x), torch.from_numpy(train_set_y)),
                            (torch.from_numpy(valid_set_x), torch.from_numpy(valid_set_y)),
                            (torch.from_numpy(test_set_x), torch.from_numpy(test_set_y)),
                            (y_mean, y_std)]
                    return rval
                #加载数据
                dataset =load_raw_data(wx_name)
                dataset_train_x.append(dataset[0][0]) # 其中 train_set_x, train_set_y = datasets[0]
                dataset_train_y.append(dataset[0][1])
                dataset_val_x.append(dataset[1][0])
                dataset_val_y.append(dataset[1][1])
                dataset_test_x.append(dataset[2][0])
                dataset_test_y.append(dataset[2][1])

                y_mean.append(dataset[3][0])
                y_std.append(dataset[3][1])
            else:
                exit('Error: unrecognized dataset')
        print()
        # 其中 dataset_train_x = [tensor1,tensor2,tensor3]


        # build model
        if args.model == 'cnn1' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'cnn2' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
        # elif args.model == 'mlp':
        #     len_in = 1
        #     for x in img_size:
        #         len_in *= x
        #     net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        elif args.model == 'cnn':
            net_glob = CNNWX().to(args.device)
        elif args.model == 'mlp':
            net_glob = MLP().to(args.device)
        elif args.model == 'LSTM':
            net_glob = LSTM().to(args.device)
        elif args.model == 'GRU':
            net_glob = GRU().to(args.device)
        elif args.model == 'RNN':
            net_glob = RNN().to(args.device)
        elif args.model == 'T':
            net_glob = Transformer().to(args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)

        net_glob.train()
        # copy weights(复制参数)
        w_glob = net_glob.state_dict()

        # training
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = float('inf')
        val_acc_list, net_list = [], []

        if args.all_clients:  #false
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]

        dict_users = 0
        for iter in range(args.epochs):
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1) #确定m最少为1
            idxs_users = np.random.choice(range(args.num_users), m, replace=False) #随机选择m个用户 2,1,3
            for i,idx in enumerate(idxs_users): # 单个客户端
                local = LocalUpdate(args=args, dataset_x=dataset_train_x, dataset_y=dataset_train_y, idxs=idx)
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            val_loss_glob = 0
            for i in range(args.num_users):
                val_loss_local, val_y_true, val_y_pred,val_time_list = validate(args, net_glob, dataset_val_x, dataset_val_y, idx=i)
                val_loss_glob += val_loss_local

            if val_loss_glob < best_loss:
                best_loss = val_loss_glob
                best_model = net_glob.state_dict()
                torch.save(best_model, './save/best_model/best_global_model_30035_29980_23233.pth')

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            logging.info('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

        # plot loss curve
        plt.figure()
        plt.plot(range(len(loss_train)), loss_train)
        plt.ylabel('train_loss')
        plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

        # testing
        #net_glob = LSTM().to(args.device)
        net_glob.load_state_dict(torch.load('./save/best_model/best_global_model_30035_29980_23233.pth'))
        net_glob.eval()


        train_loss_all = []
        train_pm_all = []

        test_loss_all = []
        test_pm_all = []
        test_y_true_all = []
        test_y_pred_all = []
        for i in [0]:#range(args.num_users):
            print(wx_list[i])
            test_loss_local, val_y_trues, val_y_preds,test_time_list = validate(args, net_glob, dataset_test_x, dataset_test_y,idx=i)
            test_pm = get_PM(val_y_preds, val_y_trues,y_mean[i], y_std[i])
            print('test_pm:',test_pm)
            logging.info('test_pm:{},weixing:{}'.format(test_pm,wx_list[i]))
            test_loss_all.append(test_loss_local)
            test_pm_all.append(test_pm)

            train_loss_local, train_y_true, train_y_pred ,train_time_list= validate(args, net_glob, dataset_train_x, dataset_train_y, idx=i)
            train_pm = get_PM(train_y_pred, train_y_true, y_mean[i], y_std[i])

            train_loss_all.append(train_loss_local)
            train_pm_all.append(train_pm)
            test_y_true_all.append( val_y_trues)
            test_y_pred_all.append(val_y_preds) 
        ##################################################画图######################################
        all_y_true = []
        all_y_pred = []
        all_residual = []
        all_time = []

        # 将真实值和预测值从 test_y_true_all 和 test_y_pred_all 中提取
        for i in range(len(val_y_trues)):
            val_y_true =  val_y_trues[i]
            val_y_pred = val_y_preds[i]
            time = test_time_list[i]  # 这里假设 date_tle 是一个包含 datetime 对象的列表
            
            # 假设 val_y_true 和 val_y_pred 是一个 1D 数组或者列表
            all_y_true.extend(val_y_true)
            all_y_pred.extend(val_y_pred)
            
            # 计算残差
            residual = np.array(val_y_true) - np.array(val_y_pred)
            all_residual.extend(residual)
            
            # 收集对应的时间
            all_time.append(time)  # 每个数据对应的时间

        # 2. 转换为 numpy 数组以便绘图
        all_y_true = np.array(all_y_true)
        print('all_y_true:',all_y_true)
        all_y_pred = np.array(all_y_pred)
        print('all_y_pred:',all_y_pred)
        all_residual = np.array(all_residual)
        #total_points = len(all_y_true)
        #x_values = np.linspace(1, 14, total_points)
        # 3. 绘制散点图
        plt.figure(figsize=(10, 6))

        # 绘制真实值（浅蓝色）
        plt.scatter(all_time, all_y_true, color='lightblue', label='True Error', s=15)

        # 绘制预测值（绿色，点稍大）
        plt.scatter(all_time, all_y_pred, color='green', label='ML_Predicted Error', s=20)  # 增大点的大小

        # 绘制残差（浅灰色）
        plt.scatter(all_time, all_residual, color='lightgray', label='Residual Error', s=15)

        # 在Y=0的水平线上画一条虚线
        plt.axhline(y=0, color='black', linestyle='--', label='Y=0 Line')


        # 设置标题和标签
        plt.xlabel('Time/d')
        plt.ylabel('Error/km')
        plt.legend() 
        #plt.yticks(np.arange(-3,4, 2))
        plt.yticks(np.arange(np.floor(min(np.random.randn(100))) - 1, np.ceil(max(np.random.randn(100))) + 1, 2))
        # 格式化时间轴为日期显示
        ticks = np.arange(-1.65, 2.55, 0.28)
        #plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.xticks(ticks=ticks, labels=np.arange(1, 16))  # 设置X轴为1到14，标签也是1到14

        plt.savefig('/root/save/ERROR_gru+FL_17795F1.png', dpi=300)  # 你可以修改文件名和路径
        # 自动调整日期格式
        plt.gcf().autofmt_xdate()

        # 显示图表
        plt.tight_layout()
        plt.show()
    
        for loss, pm in zip(train_loss_all, train_pm_all):
            print("Training loss:",loss)
            print("accuracy:", pm)
            logging.info("Training loss:%s",loss)
            logging.info("Training accuracy:%s", pm)
        #print("Training loss:{:.5f}, accuracy: {}".format(train_loss_all, train_pm_all))
        for loss, pm in zip(test_loss_all, test_pm_all):
            print("Testing loss:",test_loss_all)
            print("Testing accuracy:", test_pm_all)
            logging.info("Testing loss:%s",loss)
            logging.info("Testing accuracy:%s", pm)