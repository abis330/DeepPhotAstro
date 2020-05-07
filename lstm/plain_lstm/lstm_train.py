"""
Python package to train plain GRU-based RNN model for light curve classification
"""

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import random
import copy
import math
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

import data_utils as utils
import lstm_utils as model_utils


def append_data(list_x, list_y=None):
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

    model_utils.set_intervals(c)

    new_z = random.normalvariate(c['meta'][5], c['meta'][6] / 1.5)
    new_z = max(new_z, 0)
    new_z = min(new_z, 5)

    dt = (1 + c['meta'][5]) / (1 + new_z)
    c['meta'][5] = new_z

    # augmentation for flux
    c['hist'][:, 1] = np.random.normal(c['hist'][:, 1], c['hist'][:, 2] / 1.5)

    # multiply time intervals and wavelength to apply augmentation for red shift
    c['hist'][:, 0] *= dt
    c['hist'][:, 4] *= dt
    c['hist'][:, 5] *= dt
    c['hist'][:, 6] *= dt

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

wtable = model_utils.get_wtable(train_meta)


def mywloss(y_true, y_pred):
    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
    return loss


def get_model(X, Y, size=80):

    hist_input = Input(shape=X['hist'][0].shape, name='hist')
    meta_input = Input(shape=X['meta'][0].shape, name='meta')
    # band_input = Input(shape=X['band'][0].shape, name='band')

    # band_emb = Embedding(8, 8)(band_input)

    # hist = concatenate([hist_input]
    hist = TimeDistributed(Dense(40, activation='relu'))(hist_input)

    rnn = CuDNNLSTM(size, return_sequences=True)(hist)
    rnn = SpatialDropout1D(0.5)(rnn)

    gmp = GlobalMaxPool1D()(rnn)
    gmp = Dropout(0.5)(gmp)

    x = concatenate([meta_input, gmp])
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(15, activation='softmax')(x)

    model = Model(inputs=[hist_input, meta_input], outputs=output)

    return model


def train_model(fold_idx, samples_train, samples_valid):
    samples_train += augmentate(samples_train, utils.augment_count, utils.augment_count)
    patience = 1000000 // len(samples_train) + 5

    train_x, train_y = model_utils.get_keras_data(samples_train)
    del samples_train
    valid_x, valid_y = model_utils.get_keras_data(samples_valid)
    del samples_valid

    model = get_model(train_x, train_y)

    if fold_idx == 1:
        model.summary()
    model.compile(optimizer=model_utils.optimizer, loss=mywloss, metrics=['accuracy'])

    print('Training model {0} of {1}, Patience: {2}'.format(fold_idx, utils.num_folds, patience))
    filename = model_utils.model_filepath.format(fold_idx)
    callbacks = [EarlyStopping(patience=patience, verbose=1, monitor='val_loss'),
                 ModelCheckpoint(filename, save_best_only=True)]

    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=utils.max_epochs,
              batch_size=utils.batch_size, callbacks=callbacks, verbose=2)

    model = load_model(filename, custom_objects={'mywloss': mywloss})

    # evaluate training dataset
    train_loss, train_acc = evaluate(fold_idx, model, train_x, train_y, 'training')

    # evaluate validation dataset
    val_loss, val_acc = evaluate(fold_idx, model, valid_x, valid_y, 'validation')

    return train_loss, train_acc, val_loss, val_acc


def evaluate(fold_idx, model, x, y, datasetType):
    preds = model.predict(x, batch_size=utils.batch_size2)
    y_labels = np.argmax(y, axis=1)
    pred_labels = np.argmax(preds, axis=1)
    sess=tf.Session()
    con_mat = tf.confusion_matrix(labels=y_labels, predictions=pred_labels).eval(session=sess)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_norm = np.append(con_mat_norm, np.zeros([len(con_mat_norm),1]), 1)
    con_mat_norm = np.append(con_mat_norm, np.zeros([1, con_mat_norm.shape[1]]), 0)
    con_mat_df = pd.DataFrame(con_mat_norm, index=utils.classes, columns=utils.classes)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap='Blues')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('{} Fold {} Dataset'.format(datasetType, fold_idx))
    plt.savefig('{}_Fold_{}_confusion_matrix.png'.format(datasetType, fold_idx))
    loss = model_utils.multi_weighted_logloss(y, preds, wtable)
    acc = accuracy_score(y_labels, pred_labels)
    print('{} Fold {} MW Loss: {}, Accuracy: {}'.format(datasetType, fold_idx, loss, acc))
    true_and_preds_df = pd.DataFrame()
    true_and_preds_df['true_label'] = y_labels.tolist()
    true_and_preds_df['pred_label'] = pred_labels.tolist()
    true_and_preds_df.to_csv('lstm_{}_fold_{}_true_and_pred.csv'.format(datasetType, fold_idx), index=False)

    return loss, acc


samples = model_utils.get_data(train_data, train_meta, use_specz=utils.use_specz)

samples = np.asarray(samples)

folds = StratifiedKFold(n_splits=utils.num_folds, shuffle=True, random_state=1)
y = train_meta['target'].tolist()
train_losses = list()
train_accs = list()

val_losses = list()
val_accs = list()
for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    temp = type(val_)
    val_x = samples[val_.tolist()].tolist()
    trn_x = samples[trn_.tolist()].tolist()
    training_loss, training_acc, valid_loss, valid_acc = train_model(fold_, trn_x, val_x)
    train_losses.append(training_loss)
    train_accs.append(training_acc)
    val_losses.append(valid_loss)
    val_accs.append(valid_acc)

print('Training losses', train_losses)
print('Training accuracies', train_accs)
print('Validation losses', val_losses)
print('Validation accuracies', val_accs)
print('Mean training loss: {}'.format(np.mean(train_losses)))
print('Mean validation loss: {}'.format(np.mean(val_losses)))
print('Mean training accuracy: {}'.format(np.mean(train_accs)))
print('Mean validation accuracy: {}'.format(np.mean(val_accs)))
