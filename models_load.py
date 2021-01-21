import random

import torch
from functools import partial

import numpy as np
from torch import nn
import torchvision.models as M
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ResNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.bn=nn.BatchNorm2d(3)
        if dropout:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:

            self.net.avgpool=nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
      #  x=self.bn(x)
        return self.net(x)


class ResNetFinetune_weight(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.bn=nn.BatchNorm2d(3)
        if dropout:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:

            self.net.avgpool=nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)  # torch.Size([32, 2048, 7, 7])

        weight = x.view(x.size(0), x.size(1), -1)  # torch.Size([32, 2048, 49])
        weight = l2_normalize(weight, dim=2)
        weight = torch.bmm(torch.transpose(weight, 1, 2), weight)

        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
      #  x=self.bn(x)
        return [x, weight]

class ResNetFinetune_All(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.bn=nn.BatchNorm2d(3)
        if dropout:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:

            self.net.avgpool=nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)  # torch.Size([32, 2048, 7, 7])

        weight = x.view(x.size(0), x.size(1), -1)  # torch.Size([32, 2048, 49])
        weight = l2_normalize(weight, dim=2)
        weight = torch.bmm(torch.transpose(weight, 1, 2), weight)

        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
      #  x=self.bn(x)
        return [x, weight]
class ResNetFinetune_SpeciesLoss(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.bn=nn.BatchNorm2d(3)

        if dropout:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)

            self.species = nn.Sequential(
                nn.Dropout(),
                nn.Linear(2048, 14), #256 ,512, 1024 2048
            )
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )

        else:

            self.net.avgpool=nn.AdaptiveAvgPool2d(1)
            self.species = nn.Linear(2048, 14)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):


        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)



        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)


        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)


        # x=self.bn(x)
        x1 = self.net.fc(x)

        x2 = self.species(x)

        return x1, x2

class ResNetFinetune_SpeciesLoss_l3(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.bn=nn.BatchNorm2d(3)

        if dropout:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)

            self.species = nn.Sequential(
                nn.Dropout(),
                nn.Linear(1024, 14),
            )
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )

        else:

            self.net.avgpool=nn.AdaptiveAvgPool2d(1)
            self.species = nn.Linear(1024, 14)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):


        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)



        x = self.net.layer2(x)



        x = self.net.layer3(x)
        x2 = self.net.avgpool(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.species(x2)


        x = self.net.layer4(x)


        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)


        # x=self.bn(x)
        x1 = self.net.fc(x)


        return x1, x2

class ResNetFinetune_SpeciesLoss_weight(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.bn = nn.BatchNorm2d(3)

        if dropout:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)

            self.species = nn.Sequential(
                nn.Dropout(),
                nn.Linear(1024, 14),
            )
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )

        else:

            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
            self.species = nn.Linear(1024, 14)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)

        x = self.net.layer2(x)
        x = self.net.layer3(x)

        x2 = self.net.avgpool(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.species(x2)

        x = self.net.layer4(x)
        weight = x.view(x.size(0), x.size(1), -1)  # torch.Size([32, 2048, 49])
        weight = l2_normalize(weight, dim=2)
        weight = torch.bmm(torch.transpose(weight, 1, 2), weight)

        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)

        # x=self.bn(x)
        x1 = self.net.fc(x)

        # x2 = self.species(x2)

        return x1, x2, weight


######################################################
# loss
def l2_normalize(input, dim=2, eps=1e-8):
    input_n = torch.norm(input, dim=dim).clamp(eps).unsqueeze(dim=dim)
    return input / input_n

def weight_loss(weight):
    mask = torch.ones_like(weight)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j, j] = -1
    nw = mask * weight
    tmp, _ = torch.max(nw, dim=1)
    tmp, _ = torch.max(tmp, dim=1)
    # tmp2 = 0.0000002 * torch.sum(torch.sum(nw, dim=1), dim=1)
    loss = torch.mean(tmp)
    return loss




def weight_loss_nosum(weight):
    mask = torch.ones_like(weight)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j, j] = -1
    nw = mask * weight
    tmp, _ = torch.max(nw, dim=1)
    tmp, _ = torch.max(tmp, dim=1)
    # tmp2 = 0.0000002 * torch.sum(torch.sum(nw, dim=1), dim=1)
    loss = torch.mean(tmp)
    return loss

def weight_loss_copy(weight):
    mask = torch.ones_like(weight)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j, j] = -1
    nw = mask * weight
    tmp, _ = torch.max(nw, dim=1)
    tmp, _ = torch.max(tmp, dim=1)
    tmp2 = 0.000002 * torch.sum(torch.sum(nw, dim=1), dim=1)
    loss = torch.mean(tmp + tmp2)
    return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # smoothing = torch.Tensor(np.random.uniform(0., self.smoothing, size=[x.size(0)])).cuda()
        # confidence = 1. - smoothing

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothingCrossEntropy_Weight1(nn.Module):
    def     __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy_Weight1, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # smoothing = torch.Tensor(np.random.uniform(0., self.smoothing, size=[x.size(0)])).cuda()
        # confidence = 1. - smoothing
        xt = x[1]
        x = x[0]

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean() + weight_loss(xt)*0.1

