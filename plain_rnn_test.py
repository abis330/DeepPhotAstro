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

filename = 'model_001.hdf5'
model = load_model(filename, custom_objects={'mywloss': utils.mywloss})

chunk_test_sizes = list()
chunk_test_mw_losses = list()
chunk_test_acc = list()

idx = 0
for chunk_test_data in pd.read_csv('test_set.csv', chunksize=1000000):
    chunk_test_samples = utils.get_data(chunk_test_data, train_meta, use_specz=utils.use_specz)

    chunk_test_x, chunk_test_y = utils.get_keras_data(chunk_test_samples)
    del chunk_test_samples

    preds = model.predict(chunk_test_x, batch_size=utils.batch_size2)
    loss = utils.multi_weighted_logloss(chunk_test_y, preds, utils.wtable)
    acc = accuracy_score(np.argmax(chunk_test_y, axis=1), np.argmax(preds, axis=1))
    print('Chunk {} -> MW Loss: {.4f}, Accuracy: {.4f}'.format(idx+1, loss, acc))

    chunk_test_sizes.append(chunk_test_samples.shape[0])
    chunk_test_mw_losses.append(loss)
    chunk_test_acc.append(acc)

    idx = idx + 1

print('Total chunks processed before overall test data metric calculation: {}'.format(idx))
ctr = 0
test_acc = 0
test_mw_loss = 0
for size in chunk_test_sizes:
    test_acc = test_acc + size * chunk_test_acc[ctr]
    test_mw_loss = test_mw_loss + size * chunk_test_mw_losses[ctr]
    ctr = ctr + 1

print('Total chunks processed after overall test data metric calulcation: {}'.format(ctr))

assert idx == ctr

test_size = sum(chunk_test_sizes)
test_acc = test_acc / test_size
test_mw_loss = test_mw_loss / test_size
print('Net Test MW Loss: {.4f}, Accuracy: {.4f}'.format(test_mw_loss, test_acc))
