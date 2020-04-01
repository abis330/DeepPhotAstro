"""
Python package containing to test trained plain RNN model
"""

import plain_rnn_utils as utils
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

print('Loading test data...')

test_meta = pd.read_csv(utils.test_meta_filepath)

wtable = utils.get_wtable(test_meta, is_train=False)


def mywloss(y_true,y_pred):
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss


model = load_model(utils.model_filepath, custom_objects={'mywloss': mywloss})

test_data = pd.read_csv('test_set.csv')
test_samples = utils.get_data(test_data, test_meta, use_specz=utils.use_specz, is_train_data=False)

test_x, test_y = utils.get_keras_data(test_samples)
del test_samples

test_preds = model.predict(test_x, batch_size=utils.batch_size2)

test_labels = np.argmax(test_preds, axis=1)
con_mat = tf.confusion_matrix(labels=test_y, predictions=test_labels).numpy()

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index = utils.labels, columns = utils.classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap='Blues')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('test_confusion_matrix.png')

test_loss = utils.multi_weighted_logloss(test_y, test_preds, wtable)
test_acc = accuracy_score(np.argmax(test_y, axis=1), test_labels)
print('Test MW Loss: {.4f}, Accuracy: {.4f}'.format(test_loss, test_acc))

true_and_preds_df = pd.DataFrame()

true_and_preds_df['true_label'] = np.argmax(test_y, axis=1).tolist()
true_and_preds_df['pred_label'] = test_labels.tolist()

true_and_preds_df.to_csv('plain_rnn_true_and_pred.csv', index=False)
