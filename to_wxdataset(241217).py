
'''
1、保存单个卫星的数据格式为npy文件
'''


from matplotlib import ticker
from sgp4.api import Satrec
from sgp4.api import jday
import numpy as np
import datetime
from sgp4.earth_gravity import wgs84, wgs72
from sgp4.io import twoline2rv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import math
from collections import Counter
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import train_test_split


import gzip
import pickle
from openpyxl import Workbook, load_workbook
import re
import pandas as pd

# 下面这个函数是用来获取卫星编号的list
def get_txt_list():
    # 指定文件夹路径
    folder_path = "/root/all_tle/"
    # 获取指定文件夹下所有文件列表
    file_list = os.listdir(folder_path)
    # 遍历文件列表
    txt_list = []
    for file_name in file_list:
        # 检查文件是否以".txt"为扩展名
        if file_name.endswith(".txt"):
            # 获取编号
            txt_list.append(file_name[:-4])
    return txt_list


# 保存原始2行数据
def save_tle(s_name):
    # 第一步，将TLE数据读出来
    file_path = '/root/all_tle/' + str(s_name) +'.txt'
    # file_path = ''
    file = open(file_path, 'r')  # 'r'表示读取模式
    # 读取文件内容并将每行数据存储到列表
    lines = file.readlines()
    # 关闭文件
    file.close()

    # 第二步 获取tle数据
    save_tle = []
    save_time = []  # 这个存在的目的是删除重复的时间
    for i in range(len(lines)):
        if lines[i][0] == '1':
            Bstar = int(lines[i][53:59]) * 10 ** int(lines[i][59:61])  # 这个是获取到B*
            yeardate = lines[i][18:29]  # 这个是取日子
            if yeardate not in save_time:  # 相同时间只取一条
                save_time.append(yeardate)
                save_tle.append([yeardate, lines[i], lines[i + 1], Bstar])
    # print(save_tle)  # 数据格式：tle时间(22349.97768738), tle1, tle2, Bstar

    return save_tle

# 下面这个是通过tle数据计算出
def get_tle_time(s, t):
    # 下面是获取到时间用的
    time_satellite = twoline2rv(s, t, wgs72)
    epoch = time_satellite.epoch
    # 将epoch转换为年月日时分秒格式
    timestamp = datetime.datetime.fromtimestamp(epoch.timestamp())
    year = timestamp.year
    month = timestamp.month
    day = timestamp.day
    hour = timestamp.hour
    minute = timestamp.minute
    second = timestamp.second
    return year, month, day, hour, minute, second

def get_date_timestamp(s, t):
    # 下面是获取到时间用的
    time_satellite = twoline2rv(s, t, wgs72)
    epoch = time_satellite.epoch
    # 将epoch转换为年月日时分秒格式
    timestamp = datetime.datetime.fromtimestamp(epoch.timestamp())
    return timestamp

# 下面这个是计算日期时间差的
def cal_diff_time(year_now, month_now, day_now, hour_now, minute_now, second_now, year_pred, month_pred, day_pred,
                  hour_pred, minute_pred, second_pred):
    # 将日期时间字符串转换为datetime对象
    start_datetime = datetime.datetime(year_now, month_now, day_now, hour_now, minute_now, second_now)  # 当前时间
    end_datetime = datetime.datetime(year_pred, month_pred, day_pred, hour_pred, minute_pred, second_pred)  # 预测的时间

    # 计算日期时间差
    diff = end_datetime - start_datetime

    # 计算天数差
    days_diff = diff.days

    # 计算时分秒的差异
    seconds_diff = diff.seconds
    hours_diff = seconds_diff // 3600
    minutes_diff = (seconds_diff % 3600) // 60
    seconds_diff = seconds_diff % 60

    # 将时分秒差异转换为小数形式的天数差
    decimal_days_diff = days_diff + hours_diff / 24 + minutes_diff / (24 * 60) + seconds_diff / (24 * 3600)

    # 返回天数差
    return decimal_days_diff


# 下面这个是计算未来tle的(重新构建卫星对象)
def get_future_tle(temp_tle):
    s_temp, t_temp = temp_tle[1], temp_tle[2]
    satellite_temp = Satrec.twoline2rv(s_temp, t_temp)  # 这个是卫星建模，就是第i时刻下的卫星值
    # 下面是获取到时间用的
    year_temp, month_temp, day_temp, hour_temp, minute_temp, second_temp = get_tle_time(s_temp, t_temp)
    jd_temp, fr_temp = jday(year_temp, month_temp, day_temp, hour_temp, minute_temp, second_temp)
    # 下面这个是正确的数据
    e_temp, r_temp, v_temp = satellite_temp.sgp4(jd_temp, fr_temp)  # r这个是卫星的坐标,v是卫星的速度
    return r_temp, v_temp


# 下面是ECI转UNW的
def ECI2UNW(r, v, x):
    u = v / np.linalg.norm(v)
    tmp = np.cross(v, r)
    w = tmp / np.linalg.norm(tmp)
    n = np.cross(w, u)
    return np.dot(x, u), np.dot(x, n), np.dot(x, w)


# 去除异常值
def remove_outlier(tle_dataset, tle_dataset_y, cnt):
    # 这个是大于q3+3(q3-q1)小于q1-3(q3-q1)的被删除
    remove_list = []
    index = []
    for i, data in enumerate(tle_dataset):
        index.append([int(float(data[0])), i])

    res = cnt  # 前60天的长度
    for day in range(14):
        for i in range(3):
            selected_data = [tle_dataset_y[idx][i] for idx, data in enumerate(index) if data[0] == day]
            q1, q3 = np.percentile(selected_data, 25), np.percentile(selected_data, 75)
            IQR = (q3 - q1)
            for idx, data in enumerate(index):
                if data[0] == day:
                    if tle_dataset_y[idx][i] < q1 - 3 * IQR or tle_dataset_y[idx][i] > q3 + 3 * IQR:
                        if idx not in remove_list:
                            remove_list.append(idx)
                            if idx <= cnt:
                                res -= 1

    tle_dataset, tle_dataset_y = np.delete(tle_dataset, remove_list, axis=0), np.delete(tle_dataset_y, remove_list,
                                                                                        axis=0)
    return tle_dataset, tle_dataset_y, res


# 得到error
def get_error(tle, date_tle,s_name, start, end):
    # 1、获取tle里面的时间

    # 2、生成预测数据
    tle_dataset = []
    tle_dataset_y = []
    start_time = []
    cnt = 0

    for i in range(len(tle) - 1):
        s, t = tle[i][1], tle[i][2]
        satellite_i = Satrec.twoline2rv(s, t)  # 这个是卫星建模，就是第i时刻下的卫星值
        year_i, month_i, day_i, hour_i, minute_i, second_i  = get_tle_time(s, t)  # 这个是用tle算出来的今天的时间
        jd_i, fr_i = jday(year_i, month_i, day_i, hour_i, minute_i, second_i )
        e_i, r_i, v_i = satellite_i.sgp4(jd_i, fr_i)  # 这个是当前时刻的ECI下的坐标
        now = datetime.datetime(year_i, month_i, day_i, hour_i, minute_i, second_i)

        if now - start <= datetime.timedelta(days=61):
            t, timediff = 1, 0  # k是下一条加1，kdiff是时间差
            while timediff < 14:  # 就是时间差不超过14天
                if i + t < len(date_tle):
                    temp_s, temp_t = tle[i + t][1], tle[i + t][2]  # 下一条的数据
                    year_j, month_j, day_j, hour_j, minute_j, second_j = get_tle_time(temp_s, temp_t)
                    timediff = cal_diff_time(year_i, month_i, day_i, hour_i, minute_i, second_i, year_j, month_j, day_j,
                                             hour_j, minute_j, second_j)
                    mean_motion = float(temp_t[52:63])
                    wx_t = 1 / mean_motion
                    if timediff < 14:  # 就是时间差不超过14天
                        temp_date = get_date_timestamp(temp_s, temp_t) #下一条的数据的时间（精确到秒）
                        seg_num = 6  # 将T分割为21个点
                        for k in range(seg_num):
                            delta = datetime.timedelta(days=(wx_t / 2 - k * wx_t / seg_num))
                            tj_k = temp_date - delta  # 获取tj,k时间
                            jd_pred_k, fr_pred_k = jday(tj_k.year, tj_k.month, tj_k.day, tj_k.hour, tj_k.minute,
                                                        tj_k.second)  # 获取儒略日
                            # 获取i时刻与tj,k之间的时间差
                            timediff = cal_diff_time(year_i, month_i, day_i, hour_i, minute_i, second_i,
                                                     tj_k.year, tj_k.month, tj_k.day, tj_k.hour, tj_k.minute,
                                                     tj_k.second)

                            # 使用当前时刻预测tj,k时刻的位置
                            e_pred_k_i, r_pred_k_i, v_pred_k_i = satellite_i.sgp4(jd_pred_k, fr_pred_k)
                            # 使用下一个时刻预测tj,k时刻的位置
                            satellite_j = Satrec.twoline2rv(temp_s, temp_t)  # 这个是卫星建模，就是第j时刻下的卫星值
                            e_pred_k_j, r_pred_k_j, v_pred_k_j = satellite_j.sgp4(jd_pred_k, fr_pred_k)
                            # 输入变量
                            tle_dataset.append(
                                [timediff, r_pred_k_j[0], r_pred_k_j[1], r_pred_k_j[2], v_pred_k_j[0], v_pred_k_j[1],
                                 v_pred_k_j[2],
                                 r_i[0], r_i[1], r_i[2], v_i[0], v_i[1], v_i[2], tle[i][3]])
                            # 输出变量
                            temp_error_1, temp_error_2, temp_error_3 = ECI2UNW(r_i, v_i, [r_pred_k_i[0] - r_pred_k_j[0],
                                                                                          r_pred_k_i[1] - r_pred_k_j[1],
                                                                                          r_pred_k_i[2] - r_pred_k_j[2]])
                            tle_dataset_y.append([temp_error_1, temp_error_2, temp_error_3])
                            if now - start <= datetime.timedelta(days=60):
                                cnt += 1
                    t += 1
                else:
                    break
        else:
            break



    tle_dataset = np.asarray(tle_dataset)
    tle_dataset_y = np.asarray(tle_dataset_y)
    tle_dataset, tle_dataset_y, cnt = remove_outlier(tle_dataset, tle_dataset_y, cnt)
    start_time = np.asarray(start_time)

    x_data = tle_dataset
    y_data = tle_dataset_y
    start_time = start_time

    return x_data, y_data, start_time, date_tle, cnt

def to_standardization(x_train, y_train):
    x_train = x_train.T
    # 定义x_mean,x_std的list
    x_mean, x_std = [], []
    y_mean, y_std = [], []
    for i in range(x_train.shape[0]):
        # 求均值和方差
        x_mean.append(np.mean(x_train[i]))
        x_std.append(np.std(x_train[i]))
        if x_std[i] != 0:
            for j in range(x_train.shape[-1]):
                x_train[i][j] = (x_train[i][j] - x_mean[i]) / x_std[i]
    x_train = x_train.T

    y_train = y_train.T
    for i in range(3):
        # 求均值和方差
        y_mean.append(np.mean(y_train[i]))
        y_std.append(np.std(y_train[i]))
        if y_std[i] != 0:
            for j in range(y_train.shape[-1]):
                y_train[i][j] = (y_train[i][j] - y_mean[i]) / y_std[i]
    y_train = y_train.T

    return x_train, y_train, x_mean, x_std, y_mean, y_std


#获取全部卫星的高度的信息
def is_integer(s): #匹配整数
    return s.isdigit()
def is_date_format(string): #匹配日期
    pattern = r"\d{4}-\d{2}-\d{2}"
    match = re.fullmatch(pattern, string)
    return match is not None
def get_tle_info():
    with open('./wx_time.txt', 'r') as file:
        lines = file.readlines()
    tle_info = []
    for i in range(len(lines)):
        temp = lines[i].split()
        if len(temp) != 3:
            tle_info.append([])
            #首先第二列卫星编号加进去
            tle_info[-1].append(int(temp[1]))
            temp_flag = False
            for j in range(len(temp)):
                if is_date_format(temp[j]): #日期之前的数据都不要
                    temp_flag = True
                if temp_flag and is_integer(temp[j]): #判断整数
                    if int(temp[j]) > 10: #小于10的都不算
                        if len(tle_info[-1]) < 3:
                            tle_info[-1].append(int(temp[j]))
                        else:
                            break
    return np.asarray(tle_info)

if __name__ == '__main__':

    # # txt_list = ['55196']
    # 读取 Excel 文件
    # data_frame = pd.read_excel('D:/google downloads/L2T_loss-master/results/result(240510_cnn).xlsx',sheet_name='Sheet1')
    #
    # # 获取第一列的数据
    # txt_list = data_frame.iloc[:, 0].tolist()

    # 指定文件夹路径
    dataset_path = '/root/2021_LiBen/data/days60/0515/'
    file_list = os.listdir(dataset_path)  # 获取指定文件夹下所有文件列表
    wx_list = []

    for file_name in file_list:
        # 检查文件是否以".txt"为扩展名
        if file_name.endswith(".txt.gz"):
            # 获取编号
            wx_list.append(file_name[:-7])
 
    wx_list=[13844]

    for s_name in wx_list:


        save_tles = save_tle(s_name)

        start = datetime.datetime(2023, 1, 1, 0, 0, 0)
        end = datetime.datetime(2024, 1, 1, 0, 0, 0)

        date_tle = []
        tles = []
        for i in range(len(save_tles)):
            T_year, T_month, T_day, T_hour, T_minute, T_second = get_tle_time(save_tles[i][1],
                                                                              save_tles[i][2])  # 这个是用tle算出来的今天的时间
            date_data = datetime.datetime(T_year, T_month, T_day, T_hour, T_minute, T_second)
            if start <= date_data <= end:
                date_tle.append(date_data)
                tles.append(save_tles[i])

        if date_tle[0].month == 1 and date_tle[0].day == 1:

            if len(tles) > 200:
                print(s_name, '开始')
                # 生成误差表
                x_data1, y_data1, start_time1, date_tle1, train_shape = get_error(tles, date_tle,s_name, start, end)

                # 准备数据
                new_x_data1 = x_data1.astype(float)
                new_y_data1 = y_data1.astype(float)
                # print(new_y_data1[train_shape:])

                # 绘制误差散点图
                #show_scatter(new_x_data1[:train_shape], new_y_data1[:train_shape], s_name)
                #show_scatter(new_x_test1, new_y_test1, s_name + 'test')

                # 3、模型训练
                if new_x_data1.shape[0] > 200:

                    # # 划分训练集和测试集
                    # x_train, x_test, y_train, y_test = train_test_split(new_x_data1, new_y_data1, test_size=0.2,
                    #                                                     random_state=42)
                    feature_name = 0
                    # 标准化
                    x, y, x_mean, x_std, y_mean, y_std = to_standardization(new_x_data1, new_y_data1)
                    x_data = x[:train_shape]
                    x_test = x[train_shape:]

                    y_data = y[:train_shape]
                    y_test = y[train_shape:]

                    y_data = y[:train_shape]
                    y_test = y[train_shape:]

                    x_train, x_val = train_test_split(x_data, test_size=0.2, random_state=42)
                    y_train, y_val = train_test_split(y_data, test_size=0.2, random_state=42)


                    # 保存数据
                    data = ((x_train, y_train), (x_val, y_val), (x_test, y_test), (x_mean, x_std), (y_mean, y_std))

                    # with gzip.open('./2021_LiBen/data/days60/0515/' + str(s_name) + '.gz', 'wb') as f:
                    #     pickle.dump(data, f)
                    output_dir = '/root/2021_LiBen/data/days60/1217/'
                    os.makedirs(output_dir, exist_ok=True)
                    with open('/root/2021_LiBen/data/days60/1217/' + str(s_name) + '.pkl', 'wb') as f:
                        pickle.dump(data, f)
                        print("完成")
        else:
            pass