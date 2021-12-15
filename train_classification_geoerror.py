from __future__ import print_function
import argparse
from math import gamma
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from dataset.dataset_geoerror import DataFolder, GenerateData
from model.pointnet import PointNetClsGeoerror, feature_transform_regularizer, Loss
from model.pointnet_2 import PointNet2ClsGeoerror, Loss
import torch.nn.functional as F
from tqdm import tqdm

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument(
    '--data_path', type=str, help='file path for loading point cloud data'
)
parser.add_argument(
    '--label_path', type=str, help='file path for loading label data'
)
parser.add_argument(
    '--numpy_path', type=str, help='file path for loading data numpy'
)
parser.add_argument(
    '--cls_num', type=int, default=21, help='num classes for training'
)
parser.add_argument(
    '--lr', type=float, default=1e-4, help='initialize training learning rate'
)
parser.add_argument(
    '--validate_freq', type=int, default=10, help='indicate the frequency of validation'
)
parser.add_argument('--mode', type=str, default='train', help='indicate the mode of training or testing')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--pretrained', type=str, default='', help='model path')
parser.add_argument('--model', type=str, default='pointnet', help='pointnet or pointnet2')
parser.add_argument("--opt", type=str, default='', help='optimizer')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--drop_prob', type=float, default=0.3, help='indicate the probablity for dropout')
parser.add_argument('--step_size', type=int, default=20, help='indicate the step size for learning rate update')
parser.add_argument('--step_gamma', type=float, default=0.5, help='indicate the gamma for learning rate update')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed) 

dataset = GenerateData(opt.data_path, opt.label_path, opt.numpy_path, random_select=False)
train_dataset, test_dataset = dataset.generate_data()
train_label, test_label = dataset.generate_lable()

dataloader = torch.utils.data.DataLoader(
    DataFolder(train_dataset, train_label),
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        DataFolder(test_dataset, test_label),
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

print(len(train_dataset), len(test_dataset))
num_classes = opt.cls_num
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 实例化一个点云分类的对象
if opt.model == 'pointnet':
    classifier = PointNetClsGeoerror(k=num_classes, feature_transform=opt.feature_transform, drop_prob=opt.drop_prob)
elif opt.model == 'pointnet2':
    classifier = PointNetClsGeoerror(k=num_classes)
print('initialize a classifier')

if opt.pretrained != 'None':
    classifier.load_state_dict(torch.load(opt.pretrained))
if opt.opt == 'Adam':
    print("using Adam optimizer")
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.9, 0.999))
elif opt.opt == 'SGD':
    print("using SGD optimizer")
    optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=0.9)
else:
    raise NotImplementedError

# 控制学习率当前的设置是每20个epoch learning rate减小为原来的0.5倍
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.step_gamma)
classifier.cuda()
loss = Loss()

# 可视化损失函数以及分类的精度
num_batch = len(train_dataset) / opt.batchSize
train_acc = []
test_acc = []
epoch_list = []

def train(model, loss_func, num_batch, optimizer):
    classifier = model
    for epoch in range(opt.nepoch):
        acc_epoch = 0.
        epoch_list.append(epoch)
        scheduler.step()
        classifier = classifier.train()
        total_correct_train = 0
        total_trainset = 0
        for i, (point, label) in enumerate(dataloader, 0):
            point = point.transpose(2, 1)
            points = point.cuda()
            target = label.cuda()
            optimizer.zero_grad()
            pred, trans, trans_feat = classifier(points)        # 调用这个对象
            cls_loss = loss_func(pred, target)
            
            if opt.feature_transform:
                cls_loss += feature_transform_regularizer(trans_feat) * 0.001
            cls_loss.backward()
            optimizer.step()
            sigmoid = torch.nn.Sigmoid()
            pred_choice = sigmoid(pred).data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct_train += correct.item()
            total_trainset += point.size()[0]
            acc_epoch = acc_epoch + correct.item() / float(opt.batchSize)
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, cls_loss.item(), correct.item() / float(opt.batchSize)))
        train_acc.append(acc_epoch / (len(dataloader)))
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
        if epoch % opt.validate_freq == 0:
            acc = validate(classifier)
            test_acc.append(acc)

def validate(classifier):
    total_correct = 0
    total_testset = 0
    classifier.eval()
    for i, (point, label) in tqdm(enumerate(testdataloader, 0)):
        point = point.transpose(2, 1).cuda()
        target = label.cuda()
        pred, _, _ = classifier(point)
        sigmoid = torch.nn.Sigmoid()
        pred_choice = sigmoid(pred).data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += point.size()[0]
    test_acc = total_correct / float(total_testset)
    print("final accuracy {}".format(test_acc))
    return test_acc

def main():
    if opt.mode == 'train':
        train(classifier, loss, num_batch, optimizer)
    else:
        validate(classifier)
    
    # visualization
    plt.plot(epoch_list, train_acc, linewidth=1, marker='.', markersize=8, label='train_acc')
    plt.plot(epoch_list, test_acc, linewidth=1, marker='.', markersize=8, label='test_acc')
    plt.legend()
    plt.xlabel('epoch num')
    plt.title('{:} drop_prob {:} batchsize  {:} lr {:} epochs result'.format(opt.drop_prob, opt.batchSize, opt.lr, opt.nepoch))
    plt.savefig(opt.outf + '/vis.png')


if __name__ == '__main__':
    main()
