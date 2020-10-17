import torch.nn as nn

import torchvision.models as models


def get_classicnet(pretrained, net_name):
    if net_name == 'resnet18':
        net = models.resnet18(pretrained=pretrained)
        net.fc = nn.Linear(512, 12)
        return net
    elif net_name == 'resnet34':
        net = models.resnet18(pretrained=pretrained)
        net.fc = nn.Linear(512, 12)
        return net
    else:
        raise ValueError(f'Invalid net_name {net_name}')
