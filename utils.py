import json
import os

# from edafa import ClassPredictor
import pickle

from torch.autograd import Variable
import torch
from data_load import *
import numpy as np
from torch import nn
import torch.nn.functional as F

class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return float(self.total_value)/self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)

'''
    Disturbled
'''
class DisturbLabel(torch.nn.Module):
    def __init__(self, alpha, C):
        super(DisturbLabel, self).__init__()
        self.alpha = alpha
        self.C = C
        # Multinoulli distribution
        self.p_c = (1 - ((C - 1)/C) * (alpha/100))
        self.p_i = (1 / C) * (alpha / 100)

    def forward(self, y):
        # convert classes to index
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)

        # create disturbed labels
        depth = self.C
        y_one_hot = torch.ones(y_tensor.size()[0], depth) * self.p_i
        y_one_hot.scatter_(1, y_tensor, self.p_c)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))

        # sample from Multinoulli distribution
        distribution = torch.distributions.OneHotCategorical(y_one_hot)
        y_disturbed = distribution.sample()
        y_disturbed = y_disturbed.max(dim=1)[1]  # back to categorical

        return y_disturbed


def snapshot(savepathPre,savePath,state):

    if not os.path.exists(savepathPre):
        os.makedirs(savepathPre)
    torch.save(state, os.path.join(savepathPre, savePath))



def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))


def save_net(model, path):
    torch.save(model.state_dict(), path)
    print('[INFO] Checkpoint saved to {}'.format(path))


def load_net(model, path):
    model.load_state_dict(torch.load(path))
    print('[INFO] Checkpoint {} loaded'.format(path))
# if __name__=='__main__':
    # deleteNosiseType()
