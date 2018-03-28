import time

import numpy as np
import tensorflow as tf

from src import DataSet

#from DataSet import read_number_data_sets


train_home_dictionary = '/Users/adamzvada/Documents/School/BP/SpeechRecognition/audio_numbers'

# HYPER PARAMETERS

# mfcc
# num_features =  247
num_features = 13
num_context = 9


# Accounting the 0th index +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1
num_epoches = 100
num_hidden = 100
num_layers = 1
batch_size = 8

FIRST_INDEX = ord('a') - 1  # 0 is reserved to space



def input_placehodler(shape):
    """
        Creates tensorflow placeholder as input to network
        :param: shape of input placeholder
        :return: placehodler of given shape
    """
    # Input tensor - shape
    return tf.placeholder(tf.float32, shape, name='x_input')


def sequence_length_placehodler(shape):
    """"
        Creates tensorflow placehodler of input audio sequence length of given shape
    """
    # 1d array of size [batch_size]
    return tf.placeholder(tf.int32, shape, name="sequence_length")


def label_sparse_placehodler():
    """
        Creates tensorflow placehodler for target text label as sparse matrix.
        Sparse placeholder is needed by CTC op.
    """
    # label of data
    return tf.sparse_placeholder(tf.int32, name='y_sparse_label')


def weights_and_biases(num_hidden, num_classes):
    """"
        Creation of weights and biases for NN layer
        @:param num_hidden - input dim for weights
        @:param num_classes - output dim for weights and biases dim
        @:return (tensorflow variable weights, tensorflow variable biases)
    """

    with tf.name_scope('weights'):
        weights = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))

    with tf.name_scope('biases'):
        biases = tf.Variable(tf.constant(0., shape=[num_classes]))

    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)

    return weights, biases


def lstm_nerual_network(x_input, weights, biases, num_hidden, num_layers, sequence_placehodler):
    """

    :param weights:
    :param biases:
    :param num_hidden:
    :param num_layers:
    :param sequence_placehodler:
    :return:
    """


    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    # x_input [batch, max_time, num_classes]
    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, x_input, sequence_placehodler, dtype=tf.float32)

    # shape = tf.shape(x_input)
    # batch_s, max_time_steps = shape[0]

    # Reshaping to apply the same weights over the timesteps [batch_size*max_time, num_hidden?] - fully connected
    outputs = tf.reshape(outputs, [-1, num_hidden])

    logits = tf.add(tf.matmul(outputs, weights), biases)
    tf.summary.histogram("predictions", logits)

    # Back to original shape
    logits = tf.reshape(logits, [batch_size, -1, num_classes])

    return logits



def ctc_loss_function(logits, sparse_target, sequence_length):
    """

    :param logits:
    :param sparse_target:
    :param sequence_length:
    :return:
    """

    # Requiered by CTC [max_timesteps, batch_size, num_classes]
    logits = tf.transpose(logits, [1, 0, 2])

    loss = tf.nn.ctc_loss(sparse_target, logits, sequence_length)
    cost = tf.reduce_mean(loss)

    return cost

def network_optimizer(cost):
    """
        Defines optimizer for tensorflow graph
        :param cost: minimizing for given cost
    """

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)

    return optimizer

def ctc_decoder(logits, sequence_length):
    """

    :param logits:
    :param sequence_length:
    :return:
    """

    # Requiered by CTC [max_timesteps, batch_size, num_classes]
    logits = tf.transpose(logits, [1, 0, 2])

    decoded, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length)

    return decoded

def compute_label_error_rate(decoded_sparse_label, sparse_target):
    """

    :param decoded_sparse_label:
    :param sparse_target:
    :return:
    """
    # Compute the edit (Levenshtein) distance of the top path
    # Compute the label error rate (accuracy)
    label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded_sparse_label, tf.int32), sparse_target))

    return label_error_rate



def train_network(dataset):

    graph = tf.Graph()

    # Create tensorflow placeholders for network
    # n_input + (2 * n_input * n_context)]
    x = input_placehodler([None, None, num_features + 2*num_features*num_context])
    y_sparse = label_sparse_placehodler()
    sequence_length = sequence_length_placehodler([None])

    weights, baises = weights_and_biases(num_hidden, num_classes)

    logits = lstm_nerual_network(x, weights, baises, num_hidden, num_layers, sequence_length)

    cost = ctc_loss_function(logits, y_sparse, sequence_length)

    optimizer = network_optimizer(cost)

    decoded = ctc_decoder(logits, sequence_length)

    label_error_rate = compute_label_error_rate(decoded[0], y_sparse)


    # graph = tf.Graph()
    #
    # with graph.as_default():
    #logits, seq_len = lstm_network()
    #cost, targets, optimizer, ler, decoded = ctc_optimizer(logits, seq_len)


    # set TF logging verbosity
    #tf.logging.set_verbosity(tf.logging.INFO)

    #with tf.Session(graph=graph) as session:
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        for epoch in range(num_epoches):

            epoch_loss = 0
            ler_loss = 0

            start = time.time()

            for batch in range(int(dataset.train.num_examples / batch_size)):

                train_x, train_y_sparse, train_sequence_length = dataset.train.next_batch(batch_size)

                feed = {x : train_x, y_sparse : train_y_sparse, sequence_length : train_sequence_length}

                batch_cost, _ = session.run([cost, optimizer], feed)

                epoch_loss += batch_cost * batch_size

                ler_loss += session.run(label_error_rate, feed) * batch_size

                # Decoding
                d = session.run(decoded[0], feed_dict=feed)
                str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
                # Replacing blank label to none
                str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
                # Replacing space label to space
                str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

                print('Decoded: %s' % str_decoded)

            epoch_loss /= dataset.train.num_examples
            ler_loss /= dataset.train.num_examples

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
                  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

            print(log.format(epoch + 1, num_epoches, epoch_loss, ler_loss,
                             0, 0, time.time() - start))


if __name__ == "__main__":

    dataset = DataSet.read_number_data_sets(train_home_dictionary)

    train_network(dataset)

