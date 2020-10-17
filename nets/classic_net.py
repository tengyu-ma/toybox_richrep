import torch.nn as nn

import torchvision.models as models


def get_classicnet(pretrained, net_name):
    if net_name == 'resnet18':
        net = models.resnet18(pretrained=pretrained, num_classes=12)
    elif net_name == 'resnet34':
        net = models.resnet18(pretrained=pretrained, num_classes=12)
    elif net_name == 'inceptionv3':
        net = models.inception_v3(pretrained=pretrained, num_classes=12)
    else:
        raise ValueError(f'Invalid net_name {net_name}')
    return net
