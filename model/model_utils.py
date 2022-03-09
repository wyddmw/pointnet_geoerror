import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(classifier_name, cls_preds, label):
    batch_size = cls_preds.shape[0]
    if classifier_name == 'sigmoid':
        # apply binary cross entropy loss function
        labels = torch.zeros(cls_preds.shape, device=cls_preds.device)
        for i in range(batch_size):
            labels[i][label[i]] = 1
        # loss = (labels * (-torch.log(cls_preds))).sum() + ((1-labels) * (-torch.log(1-cls_preds))).sum()
        loss = (labels * (-torch.log(cls_preds))).sum() + ((1-labels) * (-torch.log(1-cls_preds))).sum() 
        loss = loss / batch_size
        return loss

        # labels = torch.zeros(cls_preds.shape, device=cls_preds.device)
        # for i in range(label.shape[0]):
        #     labels[i][label[i]] = 1
        # # 函数使用的multi-class binary cross entropy loss
        # loss = torch.clamp(cls_preds, min=0) - cls_preds * labels.type_as(cls_preds)
        # loss += torch.log1p(torch.exp(-torch.abs(cls_preds)))
        # loss = loss.sum() / labels.shape[0]
        # return loss

    elif classifier_name == 'softmax':
        # loss = -y_j * log(p_j) where p_j is the probablity of a softmax classifier
        # cls_index = torch.zeros(cls_preds.shape, device=cls_preds.device)
        # target = torch.zeros(cls_preds.shape, device=cls_preds.device)
        label = label.to(torch.long)
        criterion = nn.CrossEntropyLoss()
        # for i in range(batch_size):
        #     target[i][label[i]] = 1
        # loss = ((-torch.log1p(cls_preds)) * cls_index).sum() / batch_size
        loss = criterion(cls_preds, label)
        return loss
    else:
        raise NotImplementedError('wrong classifier')
        