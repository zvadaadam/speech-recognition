import tensorflow as tf


class CTCNetwork(object):
    """ Connectionist Temporal Classification (CTC) network.
    Args:
        input_size_vector (int) - dimension of input vector

    Variables:
        logits
        loss
        optimizer

    Placeholders:
        input_placeholder [batch_size, max_timestap, input_size_vector] - input for batch audio
        label_sparse_placeholder [] - label of data converted to sparse tensor
        input_seq_len_placeholder [batch_size] - length of inputs audios
    """

    def __init__(self, input_size_vector):

        self.input_size_vector = input_size_vector

        # Tesnorboard summaries
        self.summaries = []


    def generate_placeholders(self):
        """" Generates tensofflow placehodler
        Placehodlers:
            input_placeholder
            label_sparse_placeholder
            input_seq_len_placeholder
        """

        # input_x [batch_size, max_timestap, input_size_vector]
        self.input_placeholder = tf.placeholder(tf.float32, [None, None, self.input_size_vector], name='input')

        # label of input label data (y)
        self.label_sparse_placeholder = tf.sparse_placeholder(tf.int32, name='label_sparse')

        # length of input audio [batch_size]
        self.input_seq_len_placeholder = tf.placeholder(tf.int32, [None], name="sequence_length")

        # TODO: DROPOUT
        # TODO: LEARNING_RATE

        return self.input_placeholder, self.label_sparse_placeholder, self.input_seq_len_placeholder


    def loss_funtion(self):
        """ Operation for computing ctc loss.
        Returns:
            cost - operation for computing ctc loss
        """

        if self.logits is None:
            raise ValueError('Logits not defined!')

        with tf.name_scope("ctc_loss"):
            loss = tf.nn.ctc_loss(self.label_sparse_placeholder, self.logits, tf.cast(self.input_seq_len_placeholder, tf.int32))
            self.loss = tf.reduce_mean(loss, name='ctc_loss_mean')

        return self.loss

    def train_optimizer(self, optimizer=None, learning_rate=None):
        """" Training optimizer
        Args:
            optimizer (string) - enum of optimizers
            learning_rate (float) - float number for learning_rate
        """

        # TODO: gradient clipping
        # TODO: optimizer and learning_rate defined by parameters
        # TODO: minimize by set global step

        self.summaries.append(tf.summary.scalar('Loss', self.loss))

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        return self.optimizer

    def decoder(self):
        """ Operation for greedy decoding.

        """

        decoded, _ = tf.nn.ctc_greedy_decoder(self.logits, self.input_seq_len_placeholder)

        return decoded[0]

    def compute_label_error_rate(self, decoded_sparse_label):
        """ Operation for computing label error rate for our label sparse input.
        Args:
            decoded_sparse_label: operation for decoding
        Return:
            label_error_rate: operation for computing label error rate
        """
        # Compute the edit (Levenshtein) distance of the top path
        # Compute the label error rate (accuracy)
        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded_sparse_label, tf.int32), self.label_sparse_placeholder))

        self.summaries.append(tf.summary.scalar('label error rate', label_error_rate))

        return label_error_rate


