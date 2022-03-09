from __future__ import print_function
import argparse
from math import gamma
import os
import random
from unittest import TestLoader
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import datetime
import numpy as np
import time

from dataset.dataset_geoerror import DataFolder, GenerateData
from model import get_model
from model.model_utils import get_loss
from utils.utils import create_logger


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--data_path', type=str, help='file path for loading point cloud data')
    parser.add_argument('--label_path', type=str, help='file path for loading label data')
    parser.add_argument('--numpy_path', type=str, help='file path for loading data numpy')
    parser.add_argument('--cls_num', type=int, default=21, help='num classes for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='initialize training learning rate')
    parser.add_argument('--validate_freq', type=int, default=10, help='indicate the frequency of validation')
    parser.add_argument('--mode', type=str, default='train', help='indicate the mode of training or testing')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--pretrained', type=str, default='', help='model path')
    parser.add_argument('--model', type=str, default='pointnet', help='pointnet or pointnet2')
    parser.add_argument("--opt", type=str, default='', help='optimizer')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--drop_prob', type=float, default=0.3, help='indicate the probablity for dropout')
    parser.add_argument('--step_size', type=int, default=20, help='indicate the step size for learning rate update')
    parser.add_argument('--step_gamma', type=float, default=0.5, help='indicate the gamma for learning rate update')
    parser.add_argument('--classifier', type=str, default='sigmoid', choices=['sigmoid', 'softmax'], help='indicate the specific classifier')
    parser.add_argument('--activate_fn', type=str, default='relu', choices=['relu', 'sigmoid'], help='activation function')
    parser.add_argument('--normalize', action='store_true', default=False, help='whether data normalization is applied')
    parser.add_argument('--model_tag', type=str, default='baseline', help='model tag')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_config()
        # generating output dir
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    log_file = opt.outf + '/log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = create_logger(log_file)

    logger.info('*********************Start Initializing Model*********************')
    for key, val in vars(opt).items():
        logger.info('{:16} {}'.format(key, val))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    # print("Random Seed: ", opt.manualSeed)
    logger.info('Random Seed %d' % opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed) 
    # prepare dataset
    dataset = GenerateData(opt.data_path, opt.label_path, opt.numpy_path, random_select=False)
    train_dataset, test_dataset = dataset.generate_data()
    train_label, test_label = dataset.generate_lable()

    traindataloader = torch.utils.data.DataLoader(
        DataFolder(train_dataset, train_label, opt, logger),
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
            DataFolder(test_dataset, test_label, opt, logger),
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.workers))

    # model initialization
    model = get_model(opt)
    logger.info(model)

    # load pretrained model
    if opt.pretrained != 'None':
        logger.info('Loading pretrained model from %s ' % (opt.pretrained))
        model.load_state_dict(torch.load(opt.pretrained))
    model.cuda()

    # optimizer initializat
    if opt.opt == 'Adam':
        # print("using Adam optimizer")
        logger.info('Loading Adam Optimizer')
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    elif opt.opt == 'SGD':
        logger.info('Loading SGD optimizer')
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    else:
        raise NotImplementedError

    # 控制学习率当前的设置是每20个epoch learning rate减小为原来的0.5倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.step_gamma)

    num_batch = len(train_dataset) / opt.batchSize

    if opt.mode == 'train':
        logger.info('*********************Start training*********************')
        train(model, num_batch, optimizer, opt, logger, scheduler, traindataloader, testdataloader)
    else:
        logger.info('*********************Start Validating*********************')
        validate(model, testdataloader, logger)

def train(model, num_batch, optimizer, opt, logger, scheduler, traindataloader, testdataloader):
    classifier = model
    accuracy = np.zeros((opt.nepoch, 2))
    for epoch in range(opt.nepoch):
        acc_epoch = 0.
        scheduler.step()
        classifier = classifier.train()
        total_correct_train = 0
        total_trainset = 0
        for i, (point, label) in enumerate(traindataloader, 0):
            point = point.transpose(2, 1)
            points = point.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            start_time = time.time()
            pred = classifier(points)
            end_time = time.time()
            cls_loss = get_loss(opt.classifier, pred, label)
            cls_loss.backward()
            optimizer.step()
            pred_cls = pred.max(dim=-1)[1]
            correct = pred_cls.eq(label.data).sum().data.cpu().numpy()
            total_correct_train += correct
            total_trainset += pred_cls.shape[0]
            acc_epoch += correct
            logger.info('{} {}/{} train loss : {:.3f}, accuracy : {:.2f}'.format(epoch, int(i), 
                        num_batch, cls_loss.item(), correct / pred_cls.shape[0]))
        accuracy[epoch][0] = acc_epoch / total_trainset
        logger.info('Epoch %d training accuracy is %.2f' % (i+1, accuracy[epoch][0]))
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
        acc = validate(classifier, testdataloader, logger)
        accuracy[epoch][1] = acc
    np.save('{}/{}'.format(opt.outf, opt.model_tag), accuracy)
    logger.info('Saving accuracy file and finishing training')

def validate(classifier, testdataloader, logger):
    total_correct = 0
    total_testset = 0
    classifier.eval()
    total_time = 0.
    for i, (point, label) in tqdm(enumerate(testdataloader, 0)):
        point = point.transpose(2, 1).cuda()
        label = label.cuda()
        start_time = time.time()
        pred = classifier(point)
        end_time = time.time()
        total_time += end_time - start_time
        pred_choice = pred.max(dim=-1)[1]
        correct = pred_choice.eq(label.data).sum().data.cpu().numpy()
        total_correct += correct.item()
        total_testset += point.size()[0]
    test_acc = total_correct / float(total_testset)
    logger.info('Accuracy is %.2f in %.2f second' % (test_acc, total_time/total_testset))
    return test_acc

# def main():
#     if opt.mode == 'train':
#         train(model, num_batch, optimizer)
#     else:
#         validate(model)

#     # visualization
#     plt.plot(epoch_list, train_acc, linewidth=1, marker='.', markersize=8, label='train_acc')
#     plt.plot(epoch_list, test_acc, linewidth=1, marker='.', markersize=8, label='test_acc')
#     plt.legend()
#     plt.xlabel('epoch num')
#     plt.title('{:} drop_prob {:} batchsize  {:} lr {:} epochs result'.format(opt.drop_prob, opt.batchSize, opt.lr, opt.nepoch))
#     plt.savefig(opt.outf + '/vis.png')


if __name__ == '__main__':
    main()
