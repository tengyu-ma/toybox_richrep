import os
import socket

from pathlib import Path


ALL_CA = [
    'ball', 'spoon', 'mug', 'cup',
    'giraffe', 'horse', 'cat', 'duck',
    'helicopter', 'airplane', 'truck', 'car'
]

ALL_NO = list(range(1, 31))
TEST_NO = {
    'ball': [1, 7, 9],
    'spoon': [5, 7, 8],
    'mug': [12, 13, 14],
    'cup': [12, 13, 15],
    'giraffe': [1, 5, 13],
    'horse': [1, 10, 15],
    'cat': [4, 9, 15],
    'duck': [5, 9, 13],
    'helicopter': [5, 10, 15],
    'airplane': [2, 6, 15],
    'truck': [2, 6, 8],
    'car': [6, 11, 13],
}

ALL_TR = [
    'absent', 'present', 'hodgepodge',
    'rxminus', 'rxplus', 'ryminus', 'ryplus', 'rzminus', 'rzplus',
    'tx', 'ty', 'tz'
]

ALL_6FPS = list(range(1, 500))
ALL_1FPS = list(range(1, 500, 6))

ProjDir = Path(__file__).parent
CacheDir = os.path.join(ProjDir, 'exps/cache')
ToyboxMeanStdCacheFile = os.path.join(CacheDir, 'mean_std_cache.pickle')

HostName = socket.gethostname()
HostName = HostName.split('.')[-1] if '.' in HostName else HostName

ToyboxDataDirs = {
    'VUSE-103978002': '/home/mat/Data/Toybox/squares_same_nview',  # Lab Titan X, 10.20.141.250
    'ENG-AIVASLAB1': '/home/mat/Data/Toybox/squares_same_nview',  # My Lab 1060, 10.20.141.40
    'VUSE-10397': '/home/mat/Data/Toybox/squares_same_nview',  # Ryan Lab 1060, 10.20.141.186
    'tengyu-ubuntu': '/media/tengyu/DataU/Data/Toybox/squares_same_nview',  # Home
    'vampire': '/data/aivas_lab/Data/Toybox/squares_same_nview',  # ACCRE
}

ToyboxLogDirs = {
    'VUSE-103978002': '/home/mat/Log/ToyboxExp',  # Lab Titan X, 10.20.141.250
    'ENG-AIVASLAB1': '/home/mat/Log/ToyboxExp',  # My Lab 1060, 10.20.141.40
    'VUSE-10397': '/home/mat/Log/ToyboxExp',  # Ryan Lab 1060, 10.20.141.186
    'tengyu-ubuntu': '/media/tengyu/DataU/Log/ToyboxExp',  # Home
    'vampire': '/data/aivas_lab/Log/ToyboxExp',  # ACCRE
}

ToyboxDataDir = ToyboxDataDirs.get(HostName, '/content/data/squares_same_nview')  # Default Colab
ToyboxLogDir = ToyboxLogDirs.get(HostName, '/content/log/ToyboxExp')  # Default Colab

ReadCSV = False  # glob all the file locations or use the pre-saved csv dataframe file
