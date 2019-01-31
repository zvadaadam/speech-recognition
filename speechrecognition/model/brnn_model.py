import tensorflow as tf
from speechrecognition.model.rnn_model import RNNModel

class BRNNModel(RNNModel):
    """
    BRNNModel extending RNNModel for supporting bidirectional recurrent neural networks (BRNN).
    """

    def __init__(self, config, is_stack=True):
        """
        Initializer of BRNNModel object

        :param ConfigReader config: config reader object
        :param is_stack: flag for wheter to create stacked of unstacked BRNN
        """
        super(BRNNModel, self).__init__(config)

        self.is_stack = is_stack


    def build_rnn_layer(self, num_layers, num_hidden, input_placeholder, input_seq_len_placeholder, dropout_placeholder):
        """
        Method builds bidirectional rnn layer with specified params.
        Overrides function from RNNModel.

        :param int num_layers: number of layers
        :param int num_hidden: number of hidden layers
        :param tf.Placeholder input_placeholder: placeholder for audio input
        :param tf.Placeholder input_seq_len_placeholder: placeholder for max audio length
        :param int dropout_placeholder: dropout probablility placeholder
        :return: output of rnn
        """
        cells_fw = []
        cells_bw = []
        for i in range(num_layers):
            # LSTM Layer
            cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
            cell_bw= tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)

        fw_multicell = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
        bw_multicell = tf.nn.rnn_cell.MultiRNNCell(cells_bw)

        if self.is_stack:
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_multicell._cells, bw_multicell._cells, input_placeholder,
                                                                 sequence_length=input_seq_len_placeholder, dtype=tf.float32)
        else:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_multicell, cell_bw=bw_multicell, inputs=input_placeholder,
                                                          sequence_length=input_seq_len_placeholder, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)

        return outputs
