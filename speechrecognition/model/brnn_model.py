import tensorflow as tf
from speechrecognition.model.rnn_model import RNNModel

class BRNNModel(RNNModel):

    def __init__(self, config, is_stack=True):
        super(BRNNModel, self).__init__(config)

        self.is_stack = is_stack


    def build_rnn_layer(self, num_layers, num_hidden, input_placeholder, input_seq_len_placeholder, dropout_placeholder):

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
