import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from model.pointnet2_utils import PointNetSetAbstraction


class PointNet2ClsGeoerror(nn.Module):
    def __init__(self, opt):
        super(PointNet2ClsGeoerror, self).__init__()
        # in_channel = 6 if normal_channel else 3
        in_channel = 4
        num_cls = opt.cls_num
        active_fn = opt.activate_fn
        classifier = opt.classifier
        drop_prob = opt.drop_prob
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 4, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 4, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(drop_prob)
        self.fc3 = nn.Linear(256, num_cls)

        if active_fn == 'relu':
            self.active_fn = nn.ReLU()
        elif active_fn == 'sigmoid':
            self.active_fn = nn.Sigmoid()

        if classifier == 'sigmoid':
            self.classifier = nn.Sigmoid()
        elif classifier == 'softmax':
            self.classifier = nn.Softmax(dim=-1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(self.active_fn(self.bn1(self.fc1(x))))
        x = self.drop2(self.active_fn(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        prob = self.classifier(x)

        return prob


class Loss(nn.Module):
    def __init__(self, k=2):
        super(Loss, self).__init__()

    def forward(self, cls_pred, label):
        labels = torch.zeros(cls_pred.shape, device=cls_pred.device)
        for i in range(label.shape[0]):
            labels[i][label[i]] = 1
        # ???????????????multi-class binary cross entropy loss
        loss = torch.clamp(cls_pred, min=0) - cls_pred * labels.type_as(cls_pred)
        loss += torch.log1p(torch.exp(-torch.abs(cls_pred)))
        loss = loss.sum() / labels.shape[0]
        return loss


if __name__ == '__main__':
    # ???????????????model.py????????????????????????????????????????????????????????????
    # ??????batch_size??????4
    fake_input = torch.randn(4, 4, 3000)    # ?????????????????????3000???????????????????????????????????????xyz???????????????????????????
    print("Input point shape is: ", fake_input.shape)
    # ????????????????????????10??????
    classifier = PointNet2ClsGeoerror(k=10)
    pred, trans, trans_feature = classifier(fake_input) 
    print("Prediction shape is: ", pred.shape)
    # ????????????????????????????????????
    cls_prob = torch.sigmoid(pred)
    print("Probablity for each class is ", cls_prob.data.cpu().numpy())
    # ???????????????????????????????????????????????????
    pred_cls = cls_prob.max(dim=1)[1]
    print("Predicted class is: ", pred_cls)
