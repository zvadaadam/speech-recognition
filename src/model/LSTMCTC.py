import tensorflow as tf

from src.model.CTCNetwork import CTCNetwork


class LSTMCTC(CTCNetwork):
    """" Network with LSTM cells and CTC

    Args:
        num_hidden (int) - number of LSTM cells in one layer
        num_layer (int) - number of LSTM layers
        num_classes (int) - number of LSTM cells in projection layer

    """

    def __init__(self, num_hidden, num_layers, num_classes, input_size_vector):

        CTCNetwork.__init__(self, input_size_vector)

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_classes = num_classes


    def define(self):
        """" Creates Tensorflow graph model

        """

        # generates placeholders for network
        self.generate_placeholders()

        cells = []
        for _ in range(self.num_layers):

            cell = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)

            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_placeholder)

            cells.append(cell)


        stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # x_input [batch, max_time, num_classes]
        # The second output is the last state and we will no use that
        outputs, state = tf.nn.dynamic_rnn(stack, self.input_placeholder, self.input_seq_len_placeholder, dtype=tf.float32)

        # crates weights and biases for last fully connected layer
        self._weights_and_biases()

        input_shape = tf.shape(self.input_placeholder)
        batch_s, max_time_steps = input_shape[0], input_shape[1]

        # Reshaping to apply the same weights over the timesteps [batch_size*max_time, num_hidden?] - fully connected
        outputs = tf.reshape(outputs, [-1, self.num_hidden])

        # Adding fully connected layer on the end
        logits = tf.add(tf.matmul(outputs, self.weights), self.biases)
        tf.summary.histogram("predictions", logits)

        # Back to original shape
        self.logits = tf.reshape(logits, [batch_s, -1, self.num_classes])

        # convert to [max_timesteps, batch_size, num_classes]
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        return outputs, state

    def _weights_and_biases(self):
        """"
            Creation of weights and biases for NN layer
            @:param num_hidden - input dim for weights
            @:param num_classes - output dim for weights and biases dim
            @:return (tensorflow variable weights, tensorflow variable biases)
        """

        with tf.name_scope('weights'):
            self.weights = tf.Variable(tf.truncated_normal([self.num_hidden, self.num_classes], stddev=0.1))

        with tf.name_scope('biases'):
            self.biases = tf.Variable(tf.constant(0., shape=[self.num_classes]))

        self.summaries.append(tf.summary.histogram("weights", self.weights))
        self.summaries.append(tf.summary.histogram("biases", self.biases))
