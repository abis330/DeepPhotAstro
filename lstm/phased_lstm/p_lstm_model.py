import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import PhasedLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.rnn import dynamic_rnn

from p_lstm_utils import define_scope
from p_lstm_ops import MultiPRNNCell


class SequenceClassifier:

    def __init__(self, inputs, labels, num_hidden, cell_type, wtable, keep_prob=1.0,
                 sequence_length=None, learning_rate=0.001):
        self.inputs = inputs
        self.labels = labels
        self.wtable = wtable
        self.sequence_length = sequence_length
        self.num_hidden = num_hidden
        self.num_classes = int(labels.get_shape()[1])
        cells = []
        if cell_type == 'PLSTM':
            assert len(inputs) == 2, "Inputs should be a tuple of (t, x)"
            for n in num_hidden:
                cell = PhasedLSTMCell(n, use_peepholes=True)
                cell = DropoutWrapper(cell, input_keep_prob=keep_prob)
                cells.append(cell)
            self.stacked_cell = MultiPRNNCell(cells)
        elif cell_type == 'LSTM':
            for n in num_hidden:
                cell = LSTMCell(n, use_peepholes=True)
                cell = DropoutWrapper(cell, input_keep_prob=keep_prob)
                cells.append(cell)
            self.stacked_cell = MultiRNNCell(cells)
        else:
            raise ValueError('Unit {} not implemented.'.format(cell_type))
        self.learning_rate = learning_rate
        self.logits
        self.optimize
        self.accuracy

    @staticmethod
    def _last_relevant(output, length):
        # Gets the last relevant output of the RNN (equal to sequence length)
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    @define_scope
    def logits(self):
        weight = tf.Variable(tf.random_normal([self.num_hidden[-1], self.num_classes], dtype=tf.float32))
        bias = tf.Variable(tf.random_normal([self.num_classes], dtype=tf.float32))
        outputs, _ = dynamic_rnn(cell=self.stacked_cell, inputs=self.inputs,
                                 dtype=tf.float32, sequence_length=self.sequence_length)
        relevant = self._last_relevant(outputs, tf.cast(self.sequence_length, tf.int32))
        return tf.nn.bias_add(tf.matmul(relevant, weight), bias)

    @define_scope
    def loss(self):
        yc = tf.clip_by_value(self.scores, 1e-15, 1 - 1e-15)
        loss = -(tf.reduce_mean(tf.reduce_mean(tf.cast(self.labels, tf.float32) * tf.log(yc), axis=0) / self.wtable))
        return loss

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

    @define_scope
    def scores(self):
        return tf.nn.softmax(self.logits)

    @define_scope
    def predictions(self):
        return tf.argmax(self.logits, 1, output_type=tf.int32)

    @define_scope
    def actual(self):
        return tf.argmax(self.labels, 1, output_type=tf.int32)

    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(self.actual, self.predictions)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
