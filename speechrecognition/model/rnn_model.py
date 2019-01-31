import tensorflow as tf
from speechrecognition.model.base_model import BaseModel
from speechrecognition.config.config_reader import ConfigReader

class RNNModel(BaseModel):
    """
    RNNModel extending BaseModel is based on Recurrent Neural Networks and CTC loss function.
    """

    def __init__(self, config):
        """
        Initializer of RNNModel object

        :param ConfigReader config: config reader object
        """
        super(RNNModel, self).__init__(config)

        self.config = config

        # TODO: move to base_model
        self.init_placeholders(self.config.feature_size())

    def x(self):
        """NOT USED"""
        self.input_placeholder

    def y(self):
        """NOT USED"""
        self.label_sparse_placeholder

    def seq_length(self):
        """NOT USED"""
        self.input_seq_len_placeholder

    def dropout_prob(self):
        """NOT USED"""
        self.dropout_placeholder

    def loss(self):
        """NOT USED"""
        self.loss

    def optimizer(self):
        """NOT USED"""
        self.optimizer

    def decoder(self):
        """NOT USED"""
        self.decoder

    def label_error(self):
        """NOT USED"""
        self.label_error

    def build_model(self, model_inputs):
        """
        Builds Tensorflow computational graph of the network model for RNN and CTC approach.
        :param dict model_inputs: dictionary of inputs from tf.data.dataset iterator
        """

        input = model_inputs['input']
        sparse_label = model_inputs['sparse_label']
        seq_length = model_inputs['seq_length']

        num_layers = self.config.num_layers()
        num_hidden = self.config.num_hidden()
        num_classes = self.config.num_classes()

        with tf.variable_scope('model'):
            rnn_output = self.build_rnn_layer(num_layers, num_hidden, input, seq_length, self.dropout_placeholder)

            logistic_output = self.logistic_layer(rnn_output, input, num_hidden, num_classes)

        with tf.variable_scope('ctc_loss'):
            self.loss = self.ctc_loss_function(logistic_output, sparse_label, seq_length)

            self.optimizer = self.optimizer_method(self.loss)

        with tf.variable_scope('decoder'):
            self.decoder = self.ctc_decoder(logistic_output, seq_length)

            self.label_error, self.seq_error = self.label_error_rate(self.decoder, sparse_label, seq_length)


    def init_placeholders(self, feature_size):
        """
        Initialize model network placeholders
        :param int feature_size: size of feature vector
        :return:
        """

        # input_x [batch_size, max_timestap, input_size_vector]
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, None, feature_size], name='input')

        # label of input label data (y)
        self.label_sparse_placeholder = tf.sparse_placeholder(tf.int32, name='label_sparse')

        # length of input audio [batch_size]
        self.input_seq_len_placeholder = tf.placeholder(tf.int64, shape=[None], name="sequence_length")

        # dropout probability
        self.dropout_placeholder = tf.placeholder(tf.float32)



    def build_rnn_layer(self, num_layers, num_hidden, input_placeholder, input_seq_len_placeholder, dropout_placeholder):
        """
        Method builds RNN layer with specified params.
        :param int num_layers: number of layers
        :param int num_hidden: number of hidden layers
        :param tf.Placeholder input_placeholder: placeholder for audio input
        :param tf.Placeholder input_seq_len_placeholder: placeholder for max audio length
        :param int dropout_placeholder: dropout probablility placeholder
        :return: output of rnn
        """
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

    def dense_layer(self, stack_output, num_classes):
        # TODO: simplification of logistic_layer
        return tf.layers.dense(stack_output, num_classes)

    def logistic_layer(self, stack_output, input_x, num_hidden, num_classes):
        """
        Create fully connected layer
        :param stack_output: previous out layer
        :param tf.Placeholder input_x: audio input
        :param int num_hidden: number of hidden cells
        :param int num_classes: number of output classes
        :return: output layer
        """

        # return tf.contrib.layers.fully_connected(stack_output, num_classes, activation_fn=None)

        with tf.name_scope('weights'):
            self.weights = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))

        with tf.name_scope('biases'):
            self.biases = tf.Variable(tf.constant(0., shape=[num_classes]))

        input_shape = tf.shape(input_x)
        batch_s, max_time_steps = input_shape[0], input_shape[1]

        # Reshaping to apply the same weights over the timesteps [batch_size*max_time, num_hidden?] - fully connected
        outputs = tf.reshape(stack_output, [-1, num_hidden])

        # Adding fully connected layer on the end
        logits = tf.add(tf.matmul(outputs, self.weights), self.biases)

        # Back to original shape
        self.logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # convert to [max_timesteps, batch_size, num_classes]
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        return self.logits



    def ctc_loss_function(self, stack_output, sparse_label, seq_length):
        """
        Computes the CTC (Connectionist Temporal Classification) Loss.
        :param stack_output: previous out layer
        :param tf.sparse_placeholder_label: transcript label as sparse placeholder
        :param tf.placeholder seq_length: length of audio sequence
        :return: loss
        """

        with tf.name_scope("ctc_loss"):
            loss = tf.nn.ctc_loss(tf.cast(sparse_label, tf.int32), stack_output, tf.cast(seq_length, tf.int32))
            loss = tf.reduce_mean(loss, name='ctc_loss_mean')

        return loss

    def optimizer_method(self, loss):
        """
        Optimizer of the network
        :param loss: loss of network
        :return: train_step
        """

        return tf.train.AdamOptimizer().minimize(loss)

    def ctc_decoder(self, stack_output, seq_length):
        """
        Greedty decoding from the logit layer
        :param stack_output: previous out layer
        :param tf.placeholder seq_length: length of audio sequence
        :return: decoded
        """

        decoded, _ = tf.nn.ctc_greedy_decoder(stack_output, tf.cast(seq_length, tf.int32))

        return decoded[0]

    def label_error_rate(self, decoded, sparse_label, seq_length):
        """
        Operation for computing label error rate as levenshtein distance from our prediction and label
        :param decoded: decoded sparse
        :param tf.sparse_placeholder_label: transcript label as sparse placeholder
        :param tf.placeholder seq_length: length of audio sequence
        :return: label_error
        """

        label_errors = tf.edit_distance(tf.cast(decoded, tf.int32), tf.cast(sparse_label, tf.int32))

        total_label_error = tf.reduce_mean(label_errors)
        seq_errors = tf.count_nonzero(label_errors, axis=0)
        total_labels = tf.reduce_sum(seq_length)

        label_error = tf.truediv(total_label_error, tf.cast(total_labels, tf.float32), name='label_error')
        sequence_error = tf.truediv(tf.cast(seq_errors, tf.int32), tf.shape(seq_length)[0], name='sequence_error')


        return label_error, sequence_error
