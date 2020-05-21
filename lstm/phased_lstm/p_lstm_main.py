import argparse
import time
import tensorflow as tf
from p_lstm_model import SequenceClassifier
from sklearn.metrics import confusion_matrix as cm
import numpy as np
import p_lstm_train as train_utils
import data_utils as utils
import p_lstm_utils as model_utils
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import sys


def get_metrics(model, z, min_z, max_z):
    indices = tf.where(tf.logical_and(tf.greater(z, min_z), tf.less_equal(z, max_z)))
    actual = tf.gather(model.actual, indices)
    predictions = tf.gather(model.predictions, indices)
    # labels = tf.gather(model.labels, indices)
    # scores = tf.gather(model.scores, indices)
    accuracy, accuracy_op = tf.metrics.accuracy(actual, predictions)
    # TP, TP_op = tf.metrics.true_positives(actual, predictions)
    # TN, TN_op = tf.metrics.true_negatives(actual, predictions)
    # FP, FP_op = tf.metrics.false_positives(actual, predictions)
    # FN, FN_op = tf.metrics.false_negatives(actual, predictions)
    # # Precision (purity) is TP/(TP+FP)
    # precision, precision_op = tf.metrics.precision(actual, predictions)
    # # Recall (completeness, efficiency) is TP/(TP+FN)
    # recall, recall_op = tf.metrics.recall(actual, predictions)
    # F1, F1_op = 1.0/(tf.cast(TP, tf.float32) + tf.cast(FN, tf.float32))*\
    #             tf.cast(TP, tf.float32)**2.0/(tf.cast(TP, tf.float32) + 3.0*tf.cast(FP, tf.float32)), \
    #             tf.group(TP_op, TN_op, FP_op, FN_op)
    # AUC, AUC_op = tf.metrics.auc(labels, scores, num_thresholds=200)
    # metrics = accuracy # TP, TN, FP, FN, precision, recall, F1, AUC]
    # metric_update_ops =  # TP_op, TN_op, FP_op, FN_op, precision_op, recall_op, F1_op, AUC_op]
    return accuracy, accuracy_op


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', default='PLSTM', help='LSTM or PLSTM')
    parser.add_argument('--hidden', nargs='+', default=[128], help='Hidden size')
    parser.add_argument('--batch', default=1000, type=int, help='Batch size')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train for')
    parser.add_argument('--dropout', default=0, type=float, help='Dropout')
    parser.add_argument('--log', default='/content/drive/My Drive/ra_astronomy/logs/')
    parser.add_argument('-nosummary', action='store_true', help='Do not write summary log')
    args = parser.parse_args()
    num_hidden = [int(n) for n in args.hidden]
    dropout = args.dropout
    cell_type = args.cell

    print('Loading train data...')

    train_data = pd.read_csv(utils.modified_train_filepath)

    train_meta = pd.read_csv(utils.train_meta_filepath)

    wtable = model_utils.get_wtable(train_meta)

    samples = model_utils.get_data(train_data, train_meta, use_specz=utils.use_specz)

    samples = np.asarray(samples)

    folds = StratifiedKFold(n_splits=utils.num_folds, shuffle=True, random_state=1)
    y = train_meta['target'].tolist()
    train_losses = list()
    train_accs = list()

    val_losses = list()
    val_accs = list()

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        fold_train_losses = list()
        fold_train_accs = list()
        fold_val_losses = list()
        fold_val_accs = list()
        val_x = samples[val_.tolist()].tolist()
        val_dataset = train_utils.create_dataset(val_x)

        trn_x = samples[trn_.tolist()].tolist()
        trn_x += train_utils.augmentate(trn_x, utils.augment_count, utils.augment_count) # data augmentation

        train_dataset = train_utils.create_dataset(trn_x)
        train_dataset, val_dataset = train_utils.get_dataset(train_dataset, val_dataset)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        train_init_op = iterator.make_initializer(train_dataset)
        val_init_op = iterator.make_initializer(val_dataset)

        t, u_flux_values, u_flux_errors, g_flux_values, g_flux_errors, r_flux_values, r_flux_errors, i_flux_values, \
        i_flux_errors, z_flux_values, z_flux_errors, Y_flux_values, Y_flux_errors, labels, seq_length, \
        z = iterator.get_next()
        x = tf.concat([u_flux_values, u_flux_errors, g_flux_values, g_flux_errors, r_flux_values, r_flux_errors,
                       i_flux_values, i_flux_errors, z_flux_values, z_flux_errors, Y_flux_values, Y_flux_errors], 2)

        if cell_type == 'PLSTM':
            inputs = (t, x)
        else:
            inputs = x

        k_p = tf.placeholder(tf.float32)
        model = SequenceClassifier(inputs, labels, num_hidden, cell_type, wtable, sequence_length=seq_length,
                                   keep_prob=k_p)

        with tf.name_scope('metrics'):
            loss, loss_op = tf.metrics.mean(model.loss)
            accuracy, accuracy_op = get_metrics(model, z, min_z=0.0, max_z=100.0)

        metric_update_ops = [loss_op]
        metric_update_ops += [accuracy_op]#  TP_op, TN_op, FP_op, FN_op, precision_op, recall_op, F1_op, AUC_op]

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        log_file = cell_type
        log_file += '-' + '.'.join(str(n) for n in num_hidden)
        log_file += '-' + 'fold_' + str(fold_)
        log_file += '-' + str(int(time.time() * 1000))

        if not args.nosummary:
            train_writer = tf.summary.FileWriter(args.log + '/' + log_file + '/train', sess.graph)
            val_writer = tf.summary.FileWriter(args.log + '/' + log_file + '/val', sess.graph)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.scalar('TP', TP)
        # tf.summary.scalar('TN', TN)
        # tf.summary.scalar('FP', FP)
        # tf.summary.scalar('FN', FN)
        # tf.summary.scalar('precision', precision)
        # tf.summary.scalar('recall', recall)
        # tf.summary.scalar('F1', F1)
        # tf.summary.scalar('AUC', AUC)

        # AUC_SK = tf.placeholder(tf.float32)
        # tf.summary.scalar('AUC_SK', AUC_SK)

        merged = tf.summary.merge_all()
        print('All summaries merged successfully')
        best_train_loss = 0
        best_train_acc = 0
        best_val_loss = sys.float_info.max
        best_val_acc = 0
        best_actual = 0
        best_predictions = 0
        for epoch in range(args.epochs):
            # test on training data
            sess.run(tf.local_variables_initializer())
            sess.run(train_init_op)
            while True:
                try:
                    sess.run([model.optimize] + metric_update_ops, feed_dict={k_p: 1.0 - dropout})
                except tf.errors.OutOfRangeError:
                    break
            train_loss, train_acc, summary = sess.run([loss, accuracy, merged])
            if not args.nosummary:
                train_writer.add_summary(summary, epoch)

            # test on validation data
            sess.run(tf.local_variables_initializer())
            sess.run(val_init_op)
            labels, scores, actual, predictions = [], [], [], []
            while True:
                try:
                    res = sess.run([model.actual, model.predictions] + metric_update_ops, feed_dict={k_p: 1.0})
                    # labels.append(res[0])
                    # scores.append(res[1])
                    actual.append(res[0])
                    predictions.append(res[1])
                except tf.errors.OutOfRangeError:
                    break
            # labels = np.vstack(labels)
            # scores = np.vstack(scores)
            actual = np.concatenate(actual)
            predictions = np.concatenate(predictions)
            # average_auc = roc_auc_score(labels, scores, average='macro')
            val_loss, val_acc, summary = sess.run([loss, accuracy, merged], feed_dict={k_p: 1.0})
            # Plot normalized confusion matrix

            if not args.nosummary:
                val_writer.add_summary(summary, epoch)
            print('fold = {0} | epoch = {1} | train loss = {2:.3f} | train acc = {3:.3f} | val loss = {4:.3f}'
                  ' | val acc = {5:.3f}'.format(fold_, str(epoch + 1).zfill(5), train_loss, train_acc, val_loss,
                                                val_acc))
            if val_loss < best_val_loss:
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_actual = actual
                best_predictions = predictions
            fold_train_losses.append(train_loss)
            fold_train_accs.append(train_acc)
            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)

        fold_df = pd.DataFrame()
        fold_df['train_loss'] = fold_train_losses
        fold_df['train_acc'] = fold_train_accs
        fold_df['val_loss'] = fold_val_losses
        fold_df['val_acc'] = fold_val_accs

        fold_df.to_csv(model_utils.fold_metrics_filepath.format(fold_+1), index=False)

        model_utils.plot_confusion_matrix(cm(best_actual, best_predictions), classes=utils.classes, normalize=True,
                                          filename=args.log + '/' + log_file + '/val/confusion_matrix.pdf')
        train_losses.append(best_train_loss)
        train_accs.append(best_train_acc)
        val_losses.append(best_val_loss)
        val_accs.append(best_val_acc)

    print('Training losses', train_losses)
    print('Training accuracies', train_accs)
    print('Validation losses', val_losses)
    print('Validation accuracies', val_accs)
    print('Mean training loss: {}'.format(np.mean(train_losses)))
    print('Mean validation loss: {}'.format(np.mean(val_losses)))
    print('Mean training accuracy: {}'.format(np.mean(train_accs)))
    print('Mean validation accuracy: {}'.format(np.mean(val_accs)))


if __name__ == '__main__':
    main()
