import sys

# The paths are ugly because they are read from very deep, for example notebooks/Predict/Events
sys.path.append('../../../../../parlogan')
sys.path.append('../../../../../dataset-iop4230')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read a config_local if it exists just below notebook
try:
    from config_local import *
except:
    dataset_dir = '../../../../../sm2022-dataset-tpl/data'
    processed_dir = f'{dataset_dir}/processed'
    raw_dir = f'{dataset_dir}/raw'

ds_small = {
    'startTimestamp': '2021-01-01T00:00:00.000',
    'stopTimestamp' :  '2021-01-02T00:00:00.000',
    'dataset_dir'   : raw_dir,
    'processed_dir' : processed_dir
}

ds_medium = {
    'startTimestamp': '2020-12-01T00:00:00.000',
    'stopTimestamp' :  '2021-01-02T00:00:00.000',
    'dataset_dir'   : raw_dir,
    'processed_dir' : processed_dir
}

ds_big = {
    'startTimestamp': '2020-10-01T00:00:00.000',
    'stopTimestamp' :  '2021-04-01T00:00:00.000',
    'dataset_dir'   : raw_dir,
    'processed_dir' : processed_dir
}
