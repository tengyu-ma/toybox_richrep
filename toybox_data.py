import os
import conf
import math
import torch
import imageio
import numpy as np
import pandas as pd

from itertools import product
from glob import glob


class ToyboxData(torch.utils.data.Dataset):
    def __init__(self, root, tr, nview, ratio, mode, dataset, preload=True, transform=None):
        """ The Toybox dataset designed for squares_same_nview processed data

        Args:
            root: str,
                The path where you stored the squares_same_nview Toybox data
            tr: List[str], 'rzplus', 'rzminus', 'ryplus', 'ryminus', 'rxplus', 'rxminus'
                Which rotations you want to include in your train/test data
            nview: int, 18
                How many views you want to sample for each rotation, up to 18
            ratio: List[int], 25, 50, 75, 100
                Which zoom-out ratios you want to include in your train/test data
            mode: str, 'sv', 'mv', 'sp'
                Which mode you want to load your data
                'sv' - single view, 'mv' - multi view, 'sp' - sphere view
            preload: bool,
                Preload data into memory or not. In general, if the memory allows the preloading,
                training will be accelerated
            transform:
                The torch transforms after you load your data
        """
        self.root = root
        self.tr = tr
        self.nview = nview
        self.view_index = np.linspace(0, 17, nview, endpoint=True, dtype=np.int) if nview > 0 else None
        self.ratio = ratio
        self.mode = mode
        self.dataset = dataset

        self.all_files = sorted(glob(f'{root}/*/*/*/*/*.png'))
        self.all_df = self._get_df(read_csv=conf.ReadCSV)

        self.index_df = self._get_index_df()
        self.df = self._filter()
        self.transform = transform

        self.preload = preload
        self.loaded_data = self._preload() if preload else None

    def _get_df(self, read_csv):
        if read_csv:
            df = pd.read_csv(os.path.join(conf.CacheDir, 'squares_same_nview.csv'), index_col=0)
        else:
            df = pd.DataFrame(
                list(map(self._get_img_info, self.all_files)),
                columns=['path', 'dataset', 'ca', 'no', 'tr', 'fr', 'ratio'],
            )
            view_index = [i for i in range(18) for _ in range(4)] * (len(self.all_files) // 18 // 4)
            df['view_index'] = view_index
            df.to_csv(os.path.join(conf.CacheDir, 'squares_same_nview.csv'))
        return df[df.dataset == self.dataset]

    def _get_index_df(self):
        df = pd.DataFrame(
            sorted([[ca, no] for ca in conf.ALL_CA for no in conf.ALL_NO]),
            columns=['ca', 'no']
        )
        df['dataset'] = df.apply(lambda row: 'test' if row.no in conf.TEST_NO[row.ca] else 'train', axis=1)
        return df[df.dataset == self.dataset].reset_index().drop('index', axis=1)

    def _filter(self):
        df = self.all_df
        df = df[df['tr'].isin(self.tr)] if self.tr is not None else df
        df = df[df['view_index'].isin(self.view_index)] if self.view_index is not None else df
        df = df[df['ratio'].isin(self.ratio)] if self.ratio is not None else df
        return df.reset_index().drop('index', axis=1)

    def _preload(self):
        loaded_data = []
        for index in range(len(self)):
            print(f'\rLoading data... {index + 1} / {len(self)}', end='')
            loaded_data.append(self._getitem(index))
        print('')
        return loaded_data

    def _getitem(self, index):
        if self.mode == 'sv':
            info = self.df.loc[index]
            label = conf.ALL_CA.index(info.ca)

            img = imageio.imread(info.path)
            if self.transform is not None:
                img = self.transform(img)
            path = info.path
        elif self.mode == 'rich':
            info = self.index_df.loc[index]
            label = conf.ALL_CA.index(info.ca)

            rich_views = pd.DataFrame(
                product(self.tr, self.view_index, self.ratio),
                columns=['tr', 'view_index', 'ratio']
            )
            rich_views['ca'] = info.ca
            rich_views['no'] = info.no
            rich_df = pd.merge(self.df, rich_views, on=['ca', 'no', 'tr', 'view_index', 'ratio'])
            imgs = [imageio.imread(p) for p in rich_df.path]
            if self.transform is not None:
                imgs = [self.transform(img) for img in imgs]

            img = torch.stack(imgs).float()
            path = '>'.join(rich_df.path)
        else:
            raise ValueError(f'invalid mode {self.mode}')

        return img, label, path

    def __len__(self):
        return len(self.index_df) if self.mode == 'rich' else len(self.df)

    def __getitem__(self, index):
        if self.preload:
            return self.loaded_data[index]
        return self._getitem(index)

    @staticmethod
    def _get_img_info(path):
        s = path.split('/')
        ca = s[-5]
        dataset = s[-4]
        no = s[-3]
        tr, fr = s[-2].split('_')
        ratio, _ = s[-1].split('.')
        return path, dataset, ca, int(no), tr, int(fr), int(ratio)
