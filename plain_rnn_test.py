"""
Python package containing to test trained plain RNN model
"""

import plain_rnn_utils as utils
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score

print('Loading test data...')

train_meta = pd.read_csv('training_set_metadata.csv')
test_data = pd.read_csv('test_set.csv')

test_samples = utils.get_data(test_data, train_meta, use_specz=utils.use_specz)

filename = 'model_001.hdf5'

test_x, test_y = utils.get_keras_data(test_samples)
del test_samples

model = load_model(filename, custom_objects={'mywloss': utils.mywloss})
preds = model.predict(test_x, batch_size=utils.batch_size2)
loss = utils.multi_weighted_logloss(test_y, preds, utils.wtable)
acc = accuracy_score(np.argmax(test_y, axis=1), np.argmax(preds, axis=1))
print('MW Loss: {0:.4f}, Accuracy: {1:.4f}'.format(loss, acc))
