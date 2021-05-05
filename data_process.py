import pandas as pd                   #数据分析的工具
from matplotlib import pyplot as plt  #matplotlib:绘图的套件  pyplot是有命令风格的函数集合，使一幅图像做出些许改变。
import seaborn as sns                 #seaborn:基于matlplotlib的图形可视化python包
import numpy as np                    #numpy:用于操作数组的函数
import torch
from torch.autograd import Variable
import torch.utils.data as data


column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'classify']
origin_data = pd.read_csv('car.data', names=column_names)
data_ = origin_data.copy()

caratt_dict = {'buying':['vhigh', 'high', 'med', 'low'], 'maint':['vhigh', 'high', 'med', 'low'],
            'doors':['2', '3', '4', '5more'], 'persons':['2', '4', 'more'], 'lug_boot':['small', 'med', 'big'], 
            'safety':['low', 'med', 'high']}

carlabel_dict = {'classify':['unacc', 'acc', 'good', 'vgood']}

def get_length(data):
    return len(data)

def data_transform(att_dict, raw_data, label=False):
    # 对输入的原始数据进行编码
    l = get_length(raw_data)
    data_temp = []
    array_flag = False
    for iden, content in enumerate(att_dict):
        att_list = list(att_dict[content])
        for j, att in enumerate(att_dict[content]):
            index = att_list.index(att)
            data_[content] = data_[content].replace(att, index+1 if not label else index)        # 转换为float编码
        
        temp = np.array(data_[content], dtype = np.int64 if label else np.float32)
        temp = np.reshape(temp, (l, 1))
        temp = torch.from_numpy(temp)
        if not array_flag:
            array_flag = True
            data_encoded = temp
        else:
            data_encoded = torch.cat((data_encoded, temp), dim=1)
    return data_encoded

def att_split(train_num=1400):
    input_data = data_transform(caratt_dict, origin_data, False)
    train_data = input_data[:train_num]
    test_data = input_data[train_num:]
    return train_data, test_data

def label_split(train_num=1400):
    input_data = data_transform(carlabel_dict, origin_data, True)
    train_data = input_data[:train_num]
    test_data = input_data[train_num:]
    return train_data, test_data

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
    
    # data_encoded = att_transform(caratt_dict, origin_data, False)
    train_data, test_data = att_split()
    train_label, test_label = label_split()
    print(train_label[1])
    # print(data_encoded[0])
    # print(train_data[1599])
    # print(test_data[0])
