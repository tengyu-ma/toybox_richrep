import torch
import torch.nn as nn
import util

from nets.rich_net import RichNet
from exps.trainer import ToyboxTrainer

torch.backends.cudnn.benchmark = True


def exp_main(ratios, trs, nview):
    net_name = 'richnet_resnet18'
    net = RichNet(nview_all=len(ratios) * len(trs) * nview)
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-05, weight_decay=0.0)
    loss_func = nn.CrossEntropyLoss()
    hyper_p = util.HyperP(
        lr=0.5,
        batch_size=4,
        num_workers=0,
        epochs=300,
    )
    tb_trainer = ToyboxTrainer(
        tr=trs,
        nview=nview,
        ratio=ratios,
        mode='rich',
        img_size=(128, 128),
        net=net,
        net_name=net_name,
        optimizer=optimizer,
        loss_func=loss_func,
        hyper_p=hyper_p
    )
    print(f'=== {tb_trainer.exp_name} ===')
    tb_trainer.train_test_save()


def main():
    ratios = [100]
    trs = ['rzplus', 'rzminus', 'rxplus', 'rxminus']
    nview = 12  # need to <18
    exp_main(ratios, trs, nview)


if __name__ == '__main__':
    main()
