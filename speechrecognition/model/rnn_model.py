import tensorflow as tf
from speechrecognition.base.base_model import BaseModel
from speechrecognition.config.config_reader import ConfigReader

class RNNModel(BaseModel):

    def __init__(self, config):
        super(RNNModel, self).__init__(config)

        #self.config = ConfigReader(config_path)
        self.config = config

        self.build_model()


    def x(self):
        self.input_placeholder

    def y(self):
        self.label_sparse_placeholder

    def seq_length(self):
        self.input_seq_len_placeholder

    def dropout_prob(self):
        self.dropout_placeholder

    def loss(self):
        self.loss

    def optimizer(self):
        self.optimizer

    def decoder(self):
        self.decoder

    def label_error(self):
        self.label_error

    def build_model(self):

        feature_size = self.config.feature_size()

        self.init_placeholders(feature_size)

        num_layers = self.config.num_layers()
        num_hidden = self.config.num_hidden()
        num_classes = self.config.num_classes()

        rnn_output = self.build_rnn_layer(num_layers, num_hidden, self.input_placeholder, self.input_seq_len_placeholder, self.dropout_placeholder)
        logistic_output = self.logistic_layer(rnn_output, num_classes)

        self.loss = self.ctc_loss_function(logistic_output)

        self.optimizer = self.optimizer_method(self.loss)

        self.decoder = self.ctc_decoder(logistic_output)

        self.label_error = self.label_error_rate(self.decoder)


    def init_placeholders(self, feature_size):

        # input_x [batch_size, max_timestap, input_size_vector]
        self.input_placeholder = tf.placeholder(tf.float32, [None, None, feature_size], name='input')

        # label of input label data (y)
        self.label_sparse_placeholder = tf.sparse_placeholder(tf.int32, name='label_sparse')

        # length of input audio [batch_size]
        self.input_seq_len_placeholder = tf.placeholder(tf.int32, [None], name="sequence_length")

        # dropout probability
        self.dropout_placeholder = tf.placeholder(tf.float32)



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

    def logistic_layer(self, stack_output, num_classes):

        return tf.contrib.layers.fully_connected(stack_output, num_classes, activation_fn=None)

    def ctc_loss_function(self, stack_output):

        with tf.name_scope("ctc_loss"):
            loss = tf.nn.ctc_loss(self.label_sparse_placeholder, stack_output, tf.cast(self.input_seq_len_placeholder, tf.int32))
            loss = tf.reduce_mean(loss, name='ctc_loss_mean')

        return loss

    def optimizer_method(self, loss):

        return tf.train.AdamOptimizer().minimize(loss)

    def ctc_decoder(self, stack_output):

        decoded, _ = tf.nn.ctc_greedy_decoder(stack_output, self.input_seq_len_placeholder)

        return decoded[0]

    def label_error_rate(self, decoded):

        return tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), self.label_sparse_placeholder))
