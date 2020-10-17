import argparse

from exps import classic_exp, rich_exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toybox Experiments')
    parser.add_argument('-ratios', type=str, help='The ratios for the Toybox data')
    parser.add_argument('-trs', type=str, help='The transformations for the Toybox data')
    parser.add_argument('-nview', type=str, help='How many views to be used for the Toybox data')
    parser.add_argument('-net_type', type=str, help='The net_type for training, classic or rich')
    parser.add_argument('-net_names', type=str, help='The backbone net for training')

    args = parser.parse_args()

    ratios = list(map(int, args.ratios.split(' ')))
    trs = args.trs.split(' ')
    nview = int(args.nview)
    net_type = args.net_type
    net_names = args.net_names.split(' ')

    for r in ratios:
        for net_name in net_names:
            if net_type == 'classic':
                classic_exp.exp_main(ratios=[r], trs=trs, nview=nview, net_name=net_name)
            elif net_type == 'rich':
                rich_exp.exp_main(ratios=[r], trs=trs, nview=nview)
            else:
                raise ValueError(f'Invalid net type {net_type}')
