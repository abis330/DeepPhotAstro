"""
Python package containing to test trained plain RNN model
"""

import lstm_utils as model_utils
import data_utils as utils
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from keras.utils import plot_model
import gc

print('Loading test data...')

test_meta = pd.read_csv(utils.test_meta_filepath)

wtable = model_utils.get_wtable(test_meta, is_train=False)


def mywloss(y_true, y_pred):
    yc = tf.clip_by_value(y_pred, 1e-15, 1-1e-15)
    loss = - tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0)/wtable)
    return loss


# model = load_model(model_utils.model_filepath, custom_objects={'mywloss': mywloss})
#
# test_data = pd.read_csv(utils.test_data_filepath)
# test_samples = model_utils.get_data(test_data, test_meta, use_specz=model_utils.use_specz, is_train_data=True)
#
# test_x, test_y = model_utils.get_keras_data(test_samples)
# del test_samples
#
#
# def evaluate(model, x, y, datasetType):
#     preds = model.predict(x, batch_size=model_utils.batch_size2)
#     y_labels = np.argmax(y, axis=1)
#     pred_labels = np.argmax(preds, axis=1)
#     sess=tf.Session()
#     con_mat = tf.confusion_matrix(labels=y_labels, predictions=pred_labels).eval(session=sess)
#     con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
#     con_mat_norm = np.append(con_mat_norm, np.zeros([len(con_mat_norm),1]), 1)
#     con_mat_norm = np.append(con_mat_norm, np.zeros([1, con_mat_norm.shape[1]]), 0)
#     con_mat_df = pd.DataFrame(con_mat_norm, index=model_utils.classes, columns=model_utils.classes)
#     figure = plt.figure(figsize=(8, 8))
#     sns.heatmap(con_mat_df, annot=True, cmap='Blues')
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.title('{} Dataset'.format(datasetType))
#     plt.savefig('{}_confusion_matrix.png'.format(datasetType))
#     loss = model_utils.multi_weighted_logloss(y, preds, wtable)
#     acc = accuracy_score(y_labels, pred_labels)
#     print('{} MW Loss: {}, Accuracy: {}'.format(datasetType, loss, acc))
#     true_and_preds_df = pd.DataFrame()
#     true_and_preds_df['true_label'] = y_labels.tolist()
#     true_and_preds_df['pred_label'] = pred_labels.tolist()
#     true_and_preds_df.to_csv('plain_rnn_{}_true_and_pred.csv'.format(datasetType), index=False)
#
#
# evaluate(model, test_x, test_y, 'training')
#
# plot_model(model, to_file='model.png')


def predict_chunk(df_, models):
    """
    Function to predict the test chunk data using the trained models on each of the folds when performing
    cross-validation
    :return:
    """
    test_samples = model_utils.get_data(df_, test_meta, use_specz=model_utils.use_specz, is_train_data=False)
    test_x, test_y = model_utils.get_keras_data(test_samples)
    del test_samples, test_y
    gc.collect()

    preds = None
    for model in models:
        if preds is None:
            preds = model.predict(test_x, batch_size=model_utils.batch_size2) / utils.num_folds
        else:
            preds += model.predict(test_x, batch_size=model_utils.batch_size2) / utils.num_folds

    return preds


def test():
    """
    Function to test of the models
    :return:
    """
    models = list()
    for i in range(utils.num_folds):
        models.append(load_model(model_utils.model_filepath.format(i), custom_objects={'mywloss': mywloss}))

    import time
    start = time.time()
    remain_df = None

    def the_unique(x):
        return [x[i] for i in range(len(x)) if x[i] != x[i - 1]]

    for i_c, df in enumerate(pd.read_csv(utils.test_data_filepath, chunksize=utils.test_chunksize, iterator=True)):
        unique_ids = the_unique(df['object_id'].tolist())
        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()

        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])].copy()
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)

        # Create remaining samples df
        remain_df = new_remain_df

        preds_np_arr = predict_chunk(df_=df, models=models)
        preds_df = pd.DataFrame(data=preds_np_arr)
        print('Shape of predictions: {}'.format(preds_np_arr.shape))

        if i_c == 0:
            preds_df.to_csv(utils.predictions_file, header=False, index=False, float_format='%.6f')
        else:
            preds_df.to_csv(utils.predictions_file, header=False, mode='a', index=False, float_format='%.6f')

        del preds_np_arr, preds_df
        gc.collect()

        if (i_c + 1) % 10 == 0:
            utils.get_logger().info('%15d done in %5.1f' % (utils.test_chunksize * (i_c + 1),
                                                            (time.time() - start) / 60))
            print('%15d done in %5.1f' % (utils.test_chunksize * (i_c + 1), (time.time() - start) / 60))
    # Compute last object in remain_df

    preds_np_arr = predict_chunk(df_=remain_df, models=models)
    preds_df = pd.DataFrame(data=preds_np_arr)
    preds_df.to_csv(utils.predictions_file, header=False, mode='a', index=False, float_format='%.6f')
    z = pd.read_csv(utils.predictions_file)
    z = z.groupby('object_id').mean()
    z.to_csv(utils.final_predictions_file, index=True, float_format='%.6f')


if __name__ == '__main__':
    gc.enable()
    utils.create_logger()
    try:
        test()
    except Exception:
        utils.get_logger().exception('Unexpected Exception Occurred')
        raise
