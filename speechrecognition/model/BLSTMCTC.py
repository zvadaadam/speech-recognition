import tensorflow as tf

from src.model.CTCNetwork import CTCNetwork

class BLSTMCTC(CTCNetwork):

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

        output = input
        for i in range(self.num_layers):

            lstm_fw = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)

            _initial_state_fw = lstm_fw.zero_state(batch_size, tf.float32)
            _initial_state_bw = lstm_bw.zero_state(batch_size, tf.float32)

            output, _states = tf.contrib.rnn.bidirectional_rnn(lstm_fw, lstm_bw, output, initial_state_fw=_initial_state_fw, initial_state_bw=_initial_state_bw, scope='BLSTM_' + str(i + 1))
            output = tf.concat(2, [output[0], output[1]])
        cells = []

        # TODO: BLSTM


