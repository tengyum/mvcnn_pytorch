import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, 40)

        # padding 0 to make sure the small views not shrink to 0 size
        for i, m in self.net._modules.items():
            if m.__class__.__name__ == 'MaxPool2d':
                if m.padding == 0:
                    m.padding = 1

        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view(
            (int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))  # (8,12,512,7,7)
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))

