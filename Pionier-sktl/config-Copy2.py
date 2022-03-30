import sys


sys.path.append('../../../dataset-iop4230/')
sys.path.append('../Gravity/')


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset_dir = 'data'
processed_dir = f'{dataset_dir}/processed'

ds_small = {
    'startTimestamp': '2021-01-01T00:00:00.000',
    'stopTimestamp':  '2021-01-02T00:00:00.000',
    'dataset_dir': '../../../data/raw',
    'processed_dir': '../../../data/processed'
}

ds_medium = {
    'startTimestamp': '2020-12-01T00:00:00.000',
    'stopTimestamp':  '2021-01-02T00:00:00.000',
    'dataset_dir': '../../../data/raw',
    'processed_dir': '../../../data/processed'
}
