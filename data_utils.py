import logging
import numpy as np


result_dir_path = ''
train_meta_filename = 'training_set_metadata.csv'
train_filename = 'sample_training_set.csv'
modified = 'modified_'
train_meta_filepath = result_dir_path + train_meta_filename
train_filepath = result_dir_path + train_filename
modified_train_filepath = result_dir_path + modified + train_filename
modified_train_meta_filepath = result_dir_path + modified + train_meta_filename

test_data_filepath = 'test_set.csv'
test_meta_filepath = 'plasticc_test_metadata.csv'
predictions_file = 'predictions.csv'
final_predictions_file = 'final_predictions.csv'


test_chunksize = 5000000

valid_size = 0.1
num_folds = int(100 * valid_size)

augment_count = 25
batch_size = 1000
batch_size2 = 5000
use_specz = False
max_epochs = 1000

limit = 1000000
sequence_len = 256

classes = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99], dtype='int32')
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
class_names = ['class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99']

passbands = np.array([357, 477, 621, 754, 871, 1004], dtype='float32')


def create_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('simple_lightgbm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger():
    return logging.getLogger('main')
