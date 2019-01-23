import tensorflow as tf
from speechrecognition.base.base_model import BaseModel
from speechrecognition.config.config_reader import ConfigReader

class RNNModel(BaseModel):

    def __init__(self, config_path):
        super(RNNModel, self).__init__(config_path)

        self.config = ConfigReader(config_path)

        self.build_model()


    def build_model(self):

        feature_size = self.config.feature_size()

        # ______________PLACEHOLDERS______________
        # input_x [batch_size, max_timestap, input_size_vector]
        input_placeholder = tf.placeholder(tf.float32, [None, None, feature_size], name='input')

        # label of input label data (y)
        label_sparse_placeholder = tf.sparse_placeholder(tf.int32, name='label_sparse')

        # length of input audio [batch_size]
        input_seq_len_placeholder = tf.placeholder(tf.int32, [None], name="sequence_length")

        # dropout probability
        dropout_placeholder = tf.placeholder(tf.float32)


        # ______________RNN_LAYERS______________
        num_layers = self.config.num_layers()
        num_hidden = self.config.num_hidden()
        num_classes = self.config.num_classes()

        rnn_output = self.build_rnn_layer(num_layers, num_hidden, input_placeholder, input_seq_len_placeholder, dropout_placeholder)

        # ______________FULLY_CONNECTED_LAYER______________
        logistic_output = tf.contrib.layers.fully_connected(rnn_output, num_classes, activation_fn=None)

        with tf.name_scope("ctc_loss"):
            loss = tf.nn.ctc_loss(label_sparse_placeholder, logistic_output, tf.cast(input_seq_len_placeholder, tf.int32))
            loss = tf.reduce_mean(loss, name='ctc_loss_mean')
        self.loss = loss

        self.optimizer = tf.train.AdamOptimizer().minimize(loss)

        decoded, _ = tf.nn.ctc_greedy_decoder(logistic_output, input_seq_len_placeholder)[0]
        self.decoder = decoded

        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), label_sparse_placeholder))
        self.label_error_rate = label_error_rate


    def build_rnn_layer(self, num_layers, num_hidden, input_placeholder, input_seq_len_placeholder, dropout_placeholder):

        cells = []
        for i in range(num_layers):
            # LSTM Layer
            cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
            # Add Dropout
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_placeholder)
            cells.append(cell)
        # stack all RNNs
        stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        # x_input [batch, max_time, num_classes]
        # The second output is the last state and we will no use that
        outputs, state = tf.nn.dynamic_rnn(stack, input_placeholder, input_seq_len_placeholder, dtype=tf.float32)

        return outputs