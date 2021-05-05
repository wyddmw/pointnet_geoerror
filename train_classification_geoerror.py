from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
# from pointnet.dataset import ShapeNetDataset, ModelNetDataset
# from model.csv_reader import DataFolder, GenerateData
from model.dataset_geoerror import DataFolder, GenerateData
from model.model import PointNetClsGeoerror, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


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
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

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
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
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
classifier = PointNetClsGeoerror(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(train_dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, (point, label) in enumerate(dataloader, 0):
        points = point.cuda()
        target = label.cuda()
    
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)        # 调用这个对象
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        # if i % 10 == 0:
        #     j, data = next(enumerate(testdataloader, 0))
        #     points, target = data
        #     target = target[:, 0]
        #     points = points.transpose(2, 1)
        #     points, target = points.cuda(), target.cuda()
        #     classifier = classifier.eval()
        #     pred, _, _ = classifier(points)
        #     loss = F.nll_loss(pred, target)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(target.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
