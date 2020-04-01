"""
Utility to perform pre-processing on test datasets
"""

import pandas as pd
import plain_rnn_utils as utils

test_meta_df = pd.read_csv(utils.test_meta_filepath)

test_meta_df['true_target'][~test_meta_df['true_target'].isin(utils.classes)] = utils.classes[14]

test_meta_df.to_csv('plasticc_test_meta_new.csv', index=False)
