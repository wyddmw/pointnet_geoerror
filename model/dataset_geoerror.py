import csv
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import os

# parser = argparse.ArgumentParser(description='path to load csv')
# parser.add_argument('--data_path', type=str, default='./', help='denotes the path to load training data file')
# parser.add_argument('--label_path', type=str, default='./', help='denotes the path to load training label file')
# args = parser.parse_args()


class GenerateData():
    def __init__(self, file_path, label_path, numpy_file, random_select=False):
        print('generating data')
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

        if os.path.exists(self.numpy_file):
            print('loading numpy file')
            # 如果存在numpy文件，直接进行读取，否则根据输入数据的路径重新制作数据集
            all_data = np.loadtxt(self.numpy_file, delimiter=',', dtype=np.float32)
            print(all_data.shape)

        else:
            self._file_split()
            initial_flag = False            # 表示最初第一次的数据拼接，是第一行和第二行进行拼接剩下的都是当前训练数据和下一行进行拼接
            temp_data = None
             # 不存在num偏移文件，根据路径来生成
            for file in self.csv_file:
                print('writing file ', file)
                read_data = np.loadtxt(file, delimiter=',', dtype=np.float32)
                training_data = np.transpose(read_data)    # 对输入的数据进行转置 1024*3000

                # 一个文件内的四次组合
                for i in range(3):
                    if i > 0:
                        temp_data = np.concatenate((temp_data, training_data[i+1::4]), axis=1)
                    else:
                        temp_data = np.concatenate((training_data[i::4], training_data[i+1::4]), axis=1)

                if initial_flag:
                    all_data = np.concatenate((all_data, temp_data), axis=0)
                else:
                    all_data = temp_data
                    initial_flag = True
            
            # writing to file
            print('writing numpy file')
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
        # split the dataset with a radio 4:1 
        #train_index = np.array([i for i in range(4)])
        #print(train_index)
        #print(type(train_index))

        if self.random_select:
            print("random select")

        #self.train_data = all_data[train_index::5].reshape(N, -1, 4)
        self.train_data = self.train_data.reshape(num_train, -1, 4)
        self.test_data = all_data[4::5].reshape(num_test, -1, 4)

        print("train_data shape is ", self.train_data.shape)
        print("test data shape is ", self.test_data.shape)
        return self.train_data, self.test_data
        
    def generate_lable(self):
        # 只使用
        all_label = np.loadtxt(self.label_path, delimiter=',', dtype=np.float32)
        all_label = all_label[:, 0].astype(np.int) - 1
        print(all_label.shape)
        # all_label = all_label[:, 0] - 1
        
        for i in range(4):
            if i > 0:
                self.train_label = np.concatenate((self.train_label, all_label[i::5]), axis=0)
            else:
                self.train_label = all_label[::5]
        # train_index = [i for i in range(4)]
        # self.train_label = all_label[train_index::4]
        self.test_label = all_label[4::5]

        return self.train_label, self.test_label


class DataFolder(data.Dataset):
    def __init__(self, input_data, input_label):
        super(DataFolder, self).__init__()
        self.input_data = input_data
        self.input_label = input_label

    def __getitem__(self, index):
        data_index = self.input_data[index]
        label_index = self.input_label[index]

        return data_index, label_index

    def __len__(self):
        return len(self.input_data)

if __name__ == '__main__':
    # line = get_length()
    # print(line)
    # data_transform('./wine.csv')
    # file_split(args.path)
    # data_transform(args.path)
    generate_data = GenerateData(args.data_path, args.label_path)
    # generate_data._generate_training_data()
    generate_data._generate_training_lable()
