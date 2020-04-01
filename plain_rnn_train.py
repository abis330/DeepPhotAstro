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


def append_data(list_x, list_y = None):
    X = {}
    for k in list_x[0].keys():

        list = [x[k] for x in list_x]
        X[k] = np.concatenate(list)

    if list_y is None:
        return X
    else:
        return X, np.concatenate(list_y)


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

    utils.set_intervals(c)
            
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

train_data = pd.read_csv(utils.train_filepath)

train_meta = pd.read_csv(utils.train_meta_filepath)

wtable = utils.get_wtable(train_meta)


def mywloss(y_true,y_pred):
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss


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

    samples_train += augmentate(samples_train, utils.augment_count, utils.augment_count)
    patience = 1000000 // len(samples_train) + 5

    train_x, train_y = utils.get_keras_data(samples_train)
    del samples_train
    valid_x, valid_y = utils.get_keras_data(samples_valid)
    del samples_valid

    model = get_model(train_x, train_y)

    if i == 1:
        model.summary()
    model.compile(optimizer=utils.optimizer, loss=mywloss, metrics=['accuracy'])

    print('Training model {0} of {1}, Patience: {2}'.format(i, utils.num_models, patience))
    filename = 'model_{0:03d}.hdf5'.format(i)
    callbacks = [EarlyStopping(patience=patience, verbose=1), ModelCheckpoint(filename, save_best_only=True)]

    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=utils.max_epochs, batch_size=utils.batch_size, callbacks=callbacks, verbose=2)

    model = load_model(filename, custom_objects={'mywloss': mywloss})

    preds = model.predict(valid_x, batch_size=utils.batch_size2)
    loss = utils.multi_weighted_logloss(valid_y, preds, wtable)
    acc = accuracy_score(np.argmax(valid_y, axis=1), np.argmax(preds,axis=1))
    print('MW Loss: {0:.4f}, Accuracy: {1:.4f}'.format(loss, acc))


samples = utils.get_data(train_data, train_meta, use_specz=utils.use_specz)

for i in range(1, utils.num_models+1):

    samples_train, samples_valid = train_test_split(samples, test_size=utils.valid_size, random_state=42*i)
    train_model(i, samples_train, samples_valid)

