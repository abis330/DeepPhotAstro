import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import p_lstm_utils as model_utils
import copy
import random


def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes), dtype='int32')
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y


def create_dataset(given_data):
    times = [sample['hist'][:, 0] for sample in given_data]
    times = np.asarray(times)
    times = pad_sequences(times, maxlen=model_utils.sequence_len, dtype='float32', padding='post', value=-1.)
    times = np.expand_dims(times, axis=2)

    data_u_flux_values = [sample['hist'][:, 1] for sample in given_data]
    data_u_flux_values = np.asarray(data_u_flux_values)
    data_u_flux_values = pad_sequences(data_u_flux_values, maxlen=model_utils.sequence_len, dtype='float32',
                                       padding='post', value=-1.)
    data_u_flux_values = np.expand_dims(data_u_flux_values, axis=2)

    data_u_flux_errors = [sample['hist'][:, 2] for sample in given_data]
    data_u_flux_errors = np.asarray(data_u_flux_errors)
    data_u_flux_errors = pad_sequences(data_u_flux_errors, maxlen=model_utils.sequence_len, dtype='float32',
                                       padding='post', value=-1.)
    data_u_flux_errors = np.expand_dims(data_u_flux_errors, axis=2)

    data_g_flux_values = [sample['hist'][:, 3] for sample in given_data]
    data_g_flux_values = np.asarray(data_g_flux_values)
    data_g_flux_values = pad_sequences(data_g_flux_values, maxlen=model_utils.sequence_len, dtype='float32',
                                     padding='post', value=-1.)
    data_g_flux_values = np.expand_dims(data_g_flux_values, axis=2)

    data_g_flux_errors = [sample['hist'][:, 4] for sample in given_data]
    data_g_flux_errors = np.asarray(data_g_flux_errors)
    data_g_flux_errors = pad_sequences(data_g_flux_errors, maxlen=model_utils.sequence_len, dtype='float32',
                                     padding='post', value=-1.)
    data_g_flux_errors = np.expand_dims(data_g_flux_errors, axis=2)

    data_r_flux_values = [sample['hist'][:, 5] for sample in given_data]
    data_r_flux_values = np.asarray(data_r_flux_values)
    data_r_flux_values = pad_sequences(data_r_flux_values, maxlen=model_utils.sequence_len, dtype='float32',
                                     padding='post', value=-1.)
    data_r_flux_values = np.expand_dims(data_r_flux_values, axis=2)

    data_r_flux_errors = [sample['hist'][:, 6] for sample in given_data]
    data_r_flux_errors = np.asarray(data_r_flux_errors)
    data_r_flux_errors = pad_sequences(data_r_flux_errors, maxlen=model_utils.sequence_len, dtype='float32',
                                     padding='post', value=-1.)
    data_r_flux_errors = np.expand_dims(data_r_flux_errors, axis=2)

    data_i_flux_values = [sample['hist'][:, 7] for sample in given_data]
    data_i_flux_values = np.asarray(data_i_flux_values)
    data_i_flux_values = pad_sequences(data_i_flux_values, maxlen=model_utils.sequence_len, dtype='float32',
                                     padding='post', value=-1.)
    data_i_flux_values = np.expand_dims(data_i_flux_values, axis=2)

    data_i_flux_errors = [sample['hist'][:, 8] for sample in given_data]
    data_i_flux_errors = np.asarray(data_i_flux_errors)
    data_i_flux_errors = pad_sequences(data_i_flux_errors, maxlen=model_utils.sequence_len, dtype='float32',
                                     padding='post', value=-1.)
    data_i_flux_errors = np.expand_dims(data_i_flux_errors, axis=2)

    data_z_flux_values = [sample['hist'][:, 9] for sample in given_data]
    data_z_flux_values = np.asarray(data_z_flux_values)
    data_z_flux_values = pad_sequences(data_z_flux_values, maxlen=model_utils.sequence_len, dtype='float32',
                                       padding='post', value=-1.)
    data_z_flux_values = np.expand_dims(data_z_flux_values, axis=2)

    data_z_flux_errors = [sample['hist'][:, 10] for sample in given_data]
    data_z_flux_errors = np.asarray(data_z_flux_errors)
    data_z_flux_errors = pad_sequences(data_z_flux_errors, maxlen=model_utils.sequence_len, dtype='float32',
                                       padding='post', value=-1.)
    data_z_flux_errors = np.expand_dims(data_z_flux_errors, axis=2)

    data_Y_flux_values = [sample['hist'][:, 11] for sample in given_data]
    data_Y_flux_values = np.asarray(data_Y_flux_values)
    data_Y_flux_values = pad_sequences(data_Y_flux_values, maxlen=model_utils.sequence_len, dtype='float32',
                                       padding='post', value=-1.)
    data_Y_flux_values = np.expand_dims(data_Y_flux_values, axis=2)

    data_Y_flux_errors = [sample['hist'][:, 12] for sample in given_data]
    data_Y_flux_errors = np.asarray(data_Y_flux_errors)
    data_Y_flux_errors = pad_sequences(data_Y_flux_errors, maxlen=model_utils.sequence_len, dtype='float32',
                                       padding='post', value=-1.)
    data_Y_flux_errors = np.expand_dims(data_Y_flux_errors, axis=2)

    # data_src_wavelength = [sample['hist'][:, 13] for sample in given_data]
    # data_src_wavelength = np.asarray(data_src_wavelength)
    # data_src_wavelength = pad_sequences(data_src_wavelength, maxlen=model_utils.sequence_len, dtype='float32',
    #                                     padding='post', value=-1.)
    # data_src_wavelength = np.expand_dims(data_src_wavelength, axis=2)

    # data_bands = [sample['hist'][:, 14] for sample in given_data]
    # data_bands = np.asarray(data_bands)
    # data_bands = pad_sequences(data_bands, maxlen=model_utils.sequence_len, dtype='float32', padding='post',
    #                            value=-1.)
    # data_bands = np.expand_dims(data_bands, axis=2)

    data_labels = [sample['target'] for sample in given_data]
    data_labels = np.asarray(data_labels)

    data_labels = to_categorical(data_labels, 15)

    data_seq_lengths = [len(sample['hist']) for sample in given_data]
    data_seq_lengths = np.asarray(data_seq_lengths)

    redshifts = [sample['meta'][1] for sample in given_data]
    redshifts = np.asarray(redshifts)

    return (
        times,
        # data_bands,
        data_u_flux_values,
        data_u_flux_errors,
        data_g_flux_values,
        data_g_flux_errors,
        data_r_flux_values,
        data_r_flux_errors,
        data_i_flux_values,
        data_i_flux_errors,
        data_z_flux_values,
        data_z_flux_errors,
        data_Y_flux_values,
        data_Y_flux_errors,
        # data_src_wavelength,
        data_labels,
        data_seq_lengths,
        redshifts
    )


def get_dataset(train, test, batch_size=1000, test_as_train=False):
    train_dat = tf.data.Dataset.from_tensor_slices(train)

    train_dat = train_dat.batch(batch_size)
    train_dat = train_dat.shuffle(buffer_size=100)

    if test_as_train:
        test_dat = tf.data.Dataset.from_tensor_slices(train)
    else:
        test_dat = tf.data.Dataset.from_tensor_slices(test)

    test_dat = test_dat.batch(1000)

    return train_dat, test_dat


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

    new_z = random.normalvariate(c['meta'][1], c['meta'][2] / 1.5)
    new_z = max(new_z, 0)
    new_z = min(new_z, 5)

    dt = (1 + c['meta'][1]) / (1 + new_z)
    c['meta'][1] = new_z

    # augmentation for flux
    c['hist'][:, 1] = np.random.normal(c['hist'][:, 1], c['hist'][:, 2] / 1.5)
    c['hist'][:, 3] = np.random.normal(c['hist'][:, 3], c['hist'][:, 4] / 1.5)
    c['hist'][:, 5] = np.random.normal(c['hist'][:, 5], c['hist'][:, 6] / 1.5)
    c['hist'][:, 7] = np.random.normal(c['hist'][:, 7], c['hist'][:, 8] / 1.5)
    c['hist'][:, 9] = np.random.normal(c['hist'][:, 9], c['hist'][:, 10] / 1.5)
    c['hist'][:, 11] = np.random.normal(c['hist'][:, 11], c['hist'][:, 12] / 1.5)
    # multiply time intervals to apply augmentation for red shift
    c['hist'][:, 0] *= dt
    # c['hist'][:, 13] *= dt

    return c


def augmentate(samples, gl_count, exgl_count):
    res = []
    index = 0
    for s in samples:

        index += 1

        if index % 1000 == 0:
            print('Augmenting {0}/{1}   '.format(index, len(samples)), end='\r')

        count = gl_count

        for i in range(0, count):
            res.append(copy_sample(s))

    print()
    return res
