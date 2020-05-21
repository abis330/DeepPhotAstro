"""
Python package containing common constants and functions called by scripts as part of plain RNN model
"""

import functools
import tensorflow as tf
import numpy as np
import math
import warnings
import data_utils as utils
import itertools
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
warnings.filterwarnings('ignore')

augment_count = 25
batch_size = 1000
batch_size2 = 5000
optimizer = 'nadam'
num_models = 1
use_specz = False
valid_size = 0.1
max_epochs = 1000

chunksize = 100000
limit = 1000000
sequence_len = 256

classes = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99], dtype='int32')

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

class_names = ['class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65',
               'class_67','class_88','class_90','class_92','class_95','class_99']
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1, 99: 1}

# LSST passbands (nm)  u    g    r    i    z    y
passbands = np.array([357, 477, 621, 754, 871, 1004], dtype='float32')


model_filepath = utils.result_dir_path + 'model_{}.hdf5'
fold_metrics_filepath = utils.result_dir_path + 'fold_{}_metrics_progress.csv'


def get_log_flow_diff(flux):
    flux_max = np.max(flux)
    flux_min = np.min(flux)
    flux_pow = math.log2(flux_max - flux_min)
    return flux_pow


def get_data(data_df, meta_df, extragalactic=None, use_specz=False, is_train_data=True):

    samples = []
    groups = data_df.groupby('object_id')

    for g in groups:

        id = g[0]
        sample = dict()
        sample['id'] = int(id)

        meta = meta_df.loc[meta_df['object_id'] == id]

        if extragalactic == True and float(meta['hostgal_photoz']) == 0:
            continue

        if extragalactic == False and float(meta['hostgal_photoz']) > 0:
            continue

        if is_train_data:
            if 'target' in meta:
                sample['target'] = np.where(classes == int(meta['target']))[0][0]
            else:
                sample['target'] = len(classes) - 1
        else:
            if 'true_target' in meta:
                sample['target'] = np.where(classes == int(meta['true_target']))[0][0]
            else:
                sample['target'] = len(classes) - 1

        sample['meta'] = np.zeros(6, dtype='float32')

        if is_train_data:
            sample['meta'][0] = meta['ddf']
        else:
            sample['meta'][0] = meta['ddf_bool']
        sample['meta'][1] = meta['hostgal_photoz']
        sample['meta'][2] = meta['hostgal_photoz_err']
        sample['meta'][3] = meta['mwebv']
        sample['meta'][4] = float(meta['hostgal_photoz']) > 0

        sample['specz'] = float(meta['hostgal_specz'])

        if use_specz:
            sample['meta'][1] = float(meta['hostgal_specz'])
            sample['meta'][2] = 0.0

        z = float(sample['meta'][1])

        #object_id,mjd,passband,flux,flux_err,detected
        #615,59750.4229,2,-544.810303,3.622952,1

        mjd = np.array(g[1]['mjd'],      dtype='float32')
        # band = np.array(g[1]['passband'], dtype='int32')
        u_flux = np.array(g[1]['u_flux'], dtype='float32')
        u_flux_err = np.array(g[1]['u_flux_err'], dtype='float32')
        g_flux = np.array(g[1]['g_flux'], dtype='float32')
        g_flux_err = np.array(g[1]['g_flux_err'], dtype='float32')
        r_flux = np.array(g[1]['r_flux'], dtype='float32')
        r_flux_err = np.array(g[1]['r_flux_err'], dtype='float32')
        i_flux = np.array(g[1]['i_flux'], dtype='float32')
        i_flux_err = np.array(g[1]['i_flux_err'], dtype='float32')
        z_flux = np.array(g[1]['z_flux'], dtype='float32')
        z_flux_err = np.array(g[1]['z_flux_err'], dtype='float32')
        Y_flux = np.array(g[1]['Y_flux'], dtype='float32')
        Y_flux_err = np.array(g[1]['Y_flux_err'], dtype='float32')
        # detected = np.array(g[1]['detected'], dtype='float32')

        mjd -= mjd[0]
        # mjd /= 100  # Earth time shift in day*100
        mjd /= (z + 1)  # Object time shift in day*100

        # received_wavelength = passbands[band] # Earth wavelength in nm
        # received_freq = 300000 / received_wavelength # Earth frequency in THz
        # source_wavelength = received_wavelength / (z + 1) # Object wavelength in nm

        # sample['band'] = band + 1

        sample['hist'] = np.zeros((g_flux.shape[0], 13), dtype='float32')
        sample['hist'][:, 0] = mjd
        sample['hist'][:, 1] = u_flux
        sample['hist'][:, 2] = u_flux_err
        sample['hist'][:, 3] = g_flux
        sample['hist'][:, 4] = g_flux_err
        sample['hist'][:, 5] = r_flux
        sample['hist'][:, 6] = r_flux_err
        sample['hist'][:, 7] = i_flux
        sample['hist'][:, 8] = i_flux_err
        sample['hist'][:, 9] = z_flux
        sample['hist'][:, 10] = z_flux_err
        sample['hist'][:, 11] = Y_flux
        sample['hist'][:, 12] = Y_flux_err
        # sample['hist'][:,3] = detected

        # sample['hist'][:, 13] = (source_wavelength/1000)
        # sample['hist'][:, 3] = (received_wavelength/1000)
        # sample['hist'][:, 4] = band + 1

        # set_intervals(sample)

        u_flux_pow = get_log_flow_diff(u_flux)
        sample['hist'][:, 1] /= math.pow(2, u_flux_pow)
        sample['hist'][:, 2] /= math.pow(2, u_flux_pow)

        g_flux_pow = get_log_flow_diff(g_flux)
        sample['hist'][:, 3] /= math.pow(2, g_flux_pow)
        sample['hist'][:, 4] /= math.pow(2, g_flux_pow)

        r_flux_pow = get_log_flow_diff(r_flux)
        sample['hist'][:, 5] /= math.pow(2, r_flux_pow)
        sample['hist'][:, 6] /= math.pow(2, r_flux_pow)

        i_flux_pow = get_log_flow_diff(i_flux)
        sample['hist'][:, 7] /= math.pow(2, i_flux_pow)
        sample['hist'][:, 8] /= math.pow(2, i_flux_pow)

        z_flux_pow = get_log_flow_diff(z_flux)
        sample['hist'][:, 9] /= math.pow(2, z_flux_pow)
        sample['hist'][:, 10] /= math.pow(2, z_flux_pow)

        Y_flux_pow = get_log_flow_diff(Y_flux)
        sample['hist'][:, 11] /= math.pow(2, Y_flux_pow)
        sample['hist'][:, 12] /= math.pow(2, Y_flux_pow)

        sample['meta'][5] = u_flux_pow / 10
        sample['meta'][6] = g_flux_pow / 10
        sample['meta'][7] = r_flux_pow / 10
        sample['meta'][8] = i_flux_pow / 10
        sample['meta'][9] = z_flux_pow / 10
        sample['meta'][10] = Y_flux_pow / 10

        samples.append(sample)

        if len(samples) % 1000 == 0:
            print('Converting data {0}'.format(len(samples)), end='\r')

        if len(samples) >= limit:
            break

    return samples


def get_wtable(df, is_train=True):
    if is_train:
        all_y = np.array(df['target'], dtype='int32')
    else:
        all_y = np.array(df['true_target'], dtype='int32')

    y_count = np.unique(all_y, return_counts=True)[1]

    wtable = np.ones(len(classes))

    for i in range(0, y_count.shape[0]):
        wtable[i] = y_count[i] / all_y.shape[0]

    return wtable


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          filename='confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)



# def multi_weighted_logloss(y_ohe, y_p, wtable):
#     """
#     @author olivier https://www.kaggle.com/ogrellier
#     multi logloss for PLAsTiCC challenge
#     """
#     # Normalize rows and limit y_preds to 1e-15, 1-1e-15
#     y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
#     # Transform to log
#     y_p_log = np.log(y_p)
#     # Get the log for ones, .values is used to drop the index of DataFrames
#     # Exclude class 99 for now, since there is no class99 in the training set
#     # we gave a special process for that class
#     y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
#     # Get the number of positives for each class
#     nb_pos = y_ohe.sum(axis=0).astype(float)
#     nb_pos = wtable
#
#     if nb_pos[-1] == 0:
#         nb_pos[-1] = 1
#
#     # Weight average and divide by the number of positives
#     class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
#     y_w = y_log_ones * class_arr / nb_pos
#     loss = - np.sum(y_w) / np.sum(class_arr)
#     return loss / y_ohe.shape[0]


# def get_keras_data(itemslist):
#
#     keys = itemslist[0].keys()
#     X = {
#             'id': np.array([i['id'] for i in itemslist], dtype='int32'),
#             'meta': np.array([i['meta'] for i in itemslist]),
#             'hist': pad_sequences([i['hist'] for i in itemslist], maxlen=sequence_len, dtype='float32', padding='post'),
#         }
#
#     Y = to_categorical([i['target'] for i in itemslist], num_classes=len(classes))
#
#     # X['hist'][:,:,0] = 0 # remove abs time
# #    X['hist'][:,:,1] = 0 # remove flux
# #    X['hist'][:,:,2] = 0 # remove flux err
#     X['hist'][:,:,3] = 0 # remove detected flag
#     X['hist'][:,:,4] = 0 # remove fwd intervals
#     X['hist'][:,:,5] = 0 # remove bwd intervals
# #    X['hist'][:,:,6] = 0 # remove source wavelength
#     X['hist'][:,:,7] = 0 # remove received wavelength
#
#     return X, Y
