import argparse

from exps import classic_exp, rich_exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toybox Experiments')
    parser.add_argument('-ratios', type=str, help='The ratios for the Toybox data')
    parser.add_argument('-trs', type=str, help='The transformations for the Toybox data')
    parser.add_argument('-nview', type=int, help='How many views to be used for the Toybox data')
    parser.add_argument('-net_type', type=str, help='The net_type for training, classic or rich')
    parser.add_argument('-net_names', type=str, help='The backbone net for training')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Network is pretrained or not')
    parser.set_defaults(pretrained=False)

    parser.add_argument('-batch_size', type=int, help='The batchsize of training')
    parser.add_argument('-epochs', type=int, help='The number of epochs of training')

    parser.add_argument('--preload', dest='preload', action='store_true', help='Preload data into memory or not. Would recommend if the CPU memory allowed')
    parser.set_defaults(preload=False)

    args = parser.parse_args()

    ratios = list(map(int, args.ratios.split(' ')))
    trs = args.trs.split(' ')
    nview = args.nview
    net_type = args.net_type
    net_names = args.net_names.split(' ')

    print(args)

    for net_name in net_names:
        if net_type == 'classic':
            classic_exp.exp_main(
                ratios=ratios, trs=trs, nview=nview, net_name=net_name,
                pretrained=args.pretrained, batch_size=args.batch_size, epochs=args.epochs, preload=args.preload,
            )
        elif net_type == 'rich':
            rich_exp.exp_main(ratios=ratios, trs=trs, nview=nview)
        else:
            raise ValueError(f'Invalid net type {net_type}')
