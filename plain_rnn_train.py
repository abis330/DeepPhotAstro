"""
Python package to train plain GRU-based RNN model for light curve classification
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import math
import copy
import random

from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

import tensorflow as tf

import plain_rnn_utils as utils


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# augment_count = 25
# batch_size = 1000
# batch_size2 = 5000
# optimizer = 'nadam'
# num_models = 1
# use_specz = False
# valid_size = 0.1
# max_epochs = 1000
#
# limit = 1000000
# sequence_len = 256
#
# classes = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99], dtype='int32')
# class_names = ['class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99']
# class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1, 99: 1}
#
# # LSST passbands (nm)  u    g    r    i    z    y
# passbands = np.array([357, 477, 621, 754, 871, 1004], dtype='float32')


def append_data(list_x, list_y = None):
    X = {}
    for k in list_x[0].keys():

        list = [x[k] for x in list_x]
        X[k] = np.concatenate(list)

    if list_y is None:
        return X
    else:
        return X, np.concatenate(list_y)


def get_keras_data(itemslist):

    keys = itemslist[0].keys()
    X = {
            'id': np.array([i['id'] for i in itemslist], dtype='int32'),
            'meta': np.array([i['meta'] for i in itemslist]),
            'band': pad_sequences([i['band'] for i in itemslist], maxlen=sequence_len, dtype='int32'),
            'hist': pad_sequences([i['hist'] for i in itemslist], maxlen=sequence_len, dtype='float32'),
        }

    Y = to_categorical([i['target'] for i in itemslist], num_classes=len(classes))

    X['hist'][:,:,0] = 0 # remove abs time
#    X['hist'][:,:,1] = 0 # remove flux
#    X['hist'][:,:,2] = 0 # remove flux err
    X['hist'][:,:,3] = 0 # remove detected flag
#    X['hist'][:,:,4] = 0 # remove fwd intervals
#    X['hist'][:,:,5] = 0 # remove bwd intervals
#    X['hist'][:,:,6] = 0 # remove source wavelength
    X['hist'][:,:,7] = 0 # remove received wavelength

    return X, Y


def copy_sample(s, augmentate=True):
    c = copy.deepcopy(s)

    if not augmentate:
        return c

    band = []
    hist = []

    drop_rate = 0.3

    # drop some records
    for k in range(s['band'].shape[0]):
        if random.uniform(0, 1) >= drop_rate:
            band.append(s['band'][k])
            hist.append(s['hist'][k])

    c['hist'] = np.array(hist, dtype='float32')
    c['band'] = np.array(band, dtype='int32')

    set_intervals(c)
            
    new_z = random.normalvariate(c['meta'][5], c['meta'][6] / 1.5)
    new_z = max(new_z, 0)
    new_z = min(new_z, 5)

    dt = (1 + c['meta'][5]) / (1 + new_z)
    c['meta'][5] = new_z

    # augmentation for flux
    c['hist'][:,1] = np.random.normal(c['hist'][:,1], c['hist'][:,2] / 1.5)

    # multiply time intervals and wavelength to apply augmentation for red shift
    c['hist'][:,0] *= dt
    c['hist'][:,4] *= dt
    c['hist'][:,5] *= dt
    c['hist'][:,6] *= dt

    return c


def normalize_counts(samples, wtable, augmentate):
    maxpr = np.max(wtable)
    counts = maxpr / wtable

    res = []
    index = 0
    for s in samples:

        index += 1
        print('Normalizing {0}/{1}   '.format(index, len(samples)), end='\r')

        res.append(s)
        count = int(3 * counts[s['target']]) - 1

        for i in range(0, count):
            res.append(copy_sample(s, augmentate))

    print()

    return res


def augmentate(samples, gl_count, exgl_count):

    res = []
    index = 0
    for s in samples:

        index += 1
        
        if index % 1000 == 0:
            print('Augmenting {0}/{1}   '.format(index, len(samples)), end='\r')

        count = gl_count if (s['meta'][8] == 0) else exgl_count

        for i in range(0, count):
            res.append(copy_sample(s))

    print()
    return res


print('Loading train data...')

train_meta = pd.read_csv('training_set_metadata.csv')
train_data = pd.read_csv('training_set.csv')


wtable = utils.get_wtable(train_meta)


def multi_weighted_logloss(y_ohe, y_p, wtable):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    nb_pos = wtable

    if nb_pos[-1] == 0:
        nb_pos[-1] = 1

    # Weight average and divide by the number of positives
    class_arr = np.array([utils.class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss / y_ohe.shape[0]
    

def get_model(X, Y, size=80):

    hist_input = Input(shape=X['hist'][0].shape, name='hist')
    meta_input = Input(shape=X['meta'][0].shape, name='meta')
    band_input = Input(shape=X['band'][0].shape, name='band')

    band_emb = Embedding(8, 8)(band_input)

    hist = concatenate([hist_input, band_emb])
    hist = TimeDistributed(Dense(40, activation='relu'))(hist)

    rnn = CuDNNGRU(size, return_sequences=True)(hist)
    rnn = SpatialDropout1D(0.5)(rnn)

    gmp = GlobalMaxPool1D()(rnn)
    gmp = Dropout(0.5)(gmp)

    x = concatenate([meta_input, gmp])
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(15, activation='softmax')(x)

    model = Model(inputs=[hist_input, meta_input, band_input], outputs=output)

    return model


def train_model(i, samples_train, samples_valid):

    samples_train += augmentate(samples_train, augment_count, augment_count)
    patience = 1000000 // len(samples_train) + 5

    train_x, train_y = get_keras_data(samples_train)
    del samples_train
    valid_x, valid_y = get_keras_data(samples_valid)
    del samples_valid

    model = get_model(train_x, train_y)

    if i == 1:
        model.summary()
    model.compile(optimizer=utils.optimizer, loss=utils.mywloss, metrics=['accuracy'])

    print('Training model {0} of {1}, Patience: {2}'.format(i, utils.num_models, patience))
    filename = 'model_{0:03d}.hdf5'.format(i)
    callbacks = [EarlyStopping(patience=patience, verbose=1), ModelCheckpoint(filename, save_best_only=True)]

    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=utils.max_epochs, batch_size=utils.batch_size, callbacks=callbacks, verbose=2)

    model = load_model(filename, custom_objects={'mywloss': utils.mywloss})

    preds = model.predict(valid_x, batch_size=utils.batch_size2)
    loss = multi_weighted_logloss(valid_y, preds, wtable)
    acc = accuracy_score(np.argmax(valid_y, axis=1), np.argmax(preds,axis=1))
    print('MW Loss: {0:.4f}, Accuracy: {1:.4f}'.format(loss, acc))


samples = utils.get_data(train_data, train_meta, use_specz=utils.use_specz)

for i in range(1, utils.num_models+1):

    samples_train, samples_valid = train_test_split(samples, test_size=valid_size, random_state=42*i)
    train_model(i, samples_train, samples_valid)
