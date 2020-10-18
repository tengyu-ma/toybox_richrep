import torch
import torch.nn as nn
import torchvision.models as models


class RichNet(nn.Module):
    def __init__(self, nview_all):
        super().__init__()
        self.feature = nn.Sequential(*list(models.resnet18().children())[:-1])
        self.fc = nn.Linear(512, 12)

        self.nview_all = nview_all

    def forward(self, x):
        x = self.feature(x)
        n, c, h, w = x.shape
        bs = n // self.nview_all  # real batch size
        x = x.view([bs, self.nview_all, c, h, w])
        x = torch.max(x, 1)[0].view(bs, -1)
        y = self.fc(x)

        return y
