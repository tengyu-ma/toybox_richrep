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


# class MVCNN(Model):
#     def __init__(self, name, model, nclasses=12, cnn_name='vgg11', num_views=12):
#         super(MVCNN, self).__init__(name)
#
#         self.nclasses = nclasses
#         self.num_views = num_views
#         self.use_resnet = cnn_name.startswith('resnet')
#
#         if self.use_resnet:
#             self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
#             self.net_2 = model.net.fc
#         else:
#             self.net_1 = model.net_1
#             self.net_2 = model.net_2
#
#     def forward(self, x):
#         y = self.net_1(x)
#         y = y.view(
#             1,
#         )
#         return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
