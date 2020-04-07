"""
Utility to plot confusion matrix for test data predictions
"""
import plain_rnn_utils as model_utils
import data_utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import accuracy_score


print('Loading test data...')

test_meta = pd.read_csv(utils.test_meta_filepath)

wtable = model_utils.get_wtable(test_meta, is_train=True)


def mywloss(y_true,y_pred):
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss


def evaluate(datasetType):
    preds = pd.read_csv(utils.final_predictions_file).values
    y = pd.read_csv(utils.test_data_filepath, usecols=['true_target'])
    y_labels = np.argmax(y, axis=1)
    pred_labels = np.argmax(preds, axis=1)
    sess=tf.Session()
    con_mat = tf.confusion_matrix(labels=y_labels, predictions=pred_labels)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_norm = np.append(con_mat_norm, np.zeros([len(con_mat_norm),1]), 1)
    con_mat_norm = np.append(con_mat_norm, np.zeros([1, con_mat_norm.shape[1]]), 0)
    con_mat_df = pd.DataFrame(con_mat_norm, index=model_utils.classes, columns=model_utils.classes)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap='Blues')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('{} Dataset'.format(datasetType))
    plt.savefig('{}_confusion_matrix.png'.format(datasetType))
    loss = model_utils.multi_weighted_logloss(y, preds, wtable)
    acc = accuracy_score(y_labels, pred_labels)
    print('{} MW Loss: {}, Accuracy: {}'.format(datasetType, loss, acc))
    true_and_preds_df = pd.DataFrame()
    true_and_preds_df['true_label'] = y_labels.tolist()
    true_and_preds_df['pred_label'] = pred_labels.tolist()
    true_and_preds_df.to_csv('plain_rnn_{}_true_and_pred.csv'.format(datasetType), index=False)


if __name__ == '__main__':
    evaluate('test')
