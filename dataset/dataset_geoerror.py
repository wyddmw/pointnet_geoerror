import csv
from pickle import TRUE
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import os
import torchvision.transforms as transforms

# parser = argparse.ArgumentParser(description='path to load csv')
# parser.add_argument('--data_path', type=str, default='./', help='denotes the path to load training data file')
# parser.add_argument('--label_path', type=str, default='./', help='denotes the path to load training label file')
# args = parser.parse_args()


class GenerateData():
    def __init__(self, file_path, label_path, numpy_file, random_select=False):
        # print('generating data')
        super(GenerateData, self).__init__()
        self.csv_file = []      # 存放文件名的列表
        self.label = []         # 存放训练的标签
        self.train_data = None    # 存放训练的数据
        self.test_data = None
        self.train_label = None   # 存放训练的标签
        self.test_label = None
        self.file_path = file_path
        self.label_path = label_path
        self.numpy_file = numpy_file
        self.random_select = False

    def _file_split(self):
        for csv_file in os.listdir(self.file_path):
            csv_name = os.path.join(self.file_path, csv_file)
            self.csv_file.append(csv_name)

        # 根据文件名进行顺序的整理 lambda是匿名函数，文件名列表根据下标进行排列
        self.csv_file = sorted(self.csv_file, key=lambda index: int(index.split('.')[0].split('_')[1]))
    
    def generate_data(self):
        all_data = None
        if os.path.isfile(self.numpy_file):
            # 如果存在numpy文件，直接进行读取，否则根据输入数据的路径重新制作数据集
            all_data = np.loadtxt(self.numpy_file, delimiter=',', dtype=np.float32)
        else:
            if len(os.listdir(self.file_path)) > 1:
                self._file_split()
            else:
                self.csv_file = os.listdir(self.file_path)

            initial_flag = False            # 表示最初第一次的数据拼接，是第一行和第二行进行拼接剩下的都是当前训练数据和下一行进行拼接
            temp_data = None
            # 不存在numpy文件，根据路径来生成
            for index, file in enumerate(self.csv_file):
                # print('writing file ', file)
                read_data = np.loadtxt(os.path.join(str(self.file_path), file), delimiter=',', dtype=np.float32)
                all_data = np.concatenate((all_data, read_data), axis=1) if index > 0 else read_data
                
            # writing to file
            np.savetxt(self.numpy_file, all_data, fmt='%f', delimiter=',')

        # split training data
        for i in range(4):      # 4:1的比例划分数据
            if i > 0:
                self.train_data = np.concatenate((self.train_data, all_data[i::5]), axis=0)
            else:
                self.train_data = all_data[::5]
        self.test_data = all_data[4::5]

        num_train, _ = self.train_data.shape
        num_test, _ = self.test_data.shape

        #self.train_data = all_data[train_index::5].reshape(N, -1, 4)
        data_line0 = all_data[0,0:4]
        self.train_data = self.train_data.reshape(num_train, -1, 4)
        train_data_index0 = self.train_data[0, 0]
        self.test_data = all_data[4::5].reshape(num_test, -1, 4)

        return self.train_data, self.test_data
        
    def generate_lable(self):
        all_label = np.loadtxt(self.label_path, delimiter=',', dtype=np.float32)
        all_label = all_label[:, 0].astype(np.int) - 1
        
        for i in range(4):
            if i > 0:
                self.train_label = np.concatenate((self.train_label, all_label[i::5]), axis=0)
            else:
                self.train_label = all_label[::5]

        self.test_label = all_label[4::5]

        return self.train_label, self.test_label


class DataFolder(data.Dataset):
    def __init__(self, input_data, input_label, opt, logger):
        super(DataFolder, self).__init__()
        self.input_data = input_data
        self.input_label = input_label
        self.opt = opt
        self.logger = logger

    def __normalize(self, data):
        
        max = data.max(axis=0, keepdims=True)
        min = data.min(axis=0, keepdims=True)
        data = (data - min) / (max - min)

        return (data - 0.5) / 0.5

    def __getitem__(self, index):
        # save point for visualization 
        # point_vis = self.input_data[index]
        # print(point_vis.shape)
        # np.save('point.npy', point_vis)
        if self.opt.normalize:
            self.input_data[index] = self.__normalize(self.input_data[index])
        data = self.input_data[index]
        label = self.input_label[index]
        return data, label

    def __len__(self):
        return len(self.input_data)


