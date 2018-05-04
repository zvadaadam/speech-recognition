import sys
import time

import numpy as np
import tensorflow as tf
import yaml

from src.config.ConfigReader import ConfigReader
from src.dataset import DataSet
from src.model.LSTMCTC import LSTMCTC
from src.dataset.VCTKDataset import VCTKDataset

#from DataSet import read_number_data_sets


FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


def train_network(dataset, config_reader):

    tf.logging.set_verbosity(tf.logging.INFO)

    logs_path = config_reader.get_tensorboard_logs_path()

    # Get Network parameters
    num_classes = config_reader.get_num_classes()
    num_epoches = config_reader.get_num_epoches()
    num_hidden = config_reader.get_num_hidden()
    num_layers = config_reader.get_num_layers()
    batch_size = config_reader.get_batch_size()
    num_features = config_reader.get_num_features()
    num_context = config_reader.get_num_context()


    graph = tf.Graph()

    lstm_ctc = LSTMCTC(num_hidden, num_layers, num_classes, num_features)
    lstm_ctc.define()
    loss_operation = lstm_ctc.loss_funtion()
    optimizer_operation = lstm_ctc.train_optimizer()
    decode_operation = lstm_ctc.decoder()
    ler_operation = lstm_ctc.compute_label_error_rate(decode_operation)

    # set TF logging verbosity
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(logs_path, graph=session.graph)

        for epoch in range(num_epoches):

            epoch_loss = 0
            ler_loss = 0

            start = time.time()

            current_state = np.zeros((num_layers, 2, batch_size, num_hidden))

            #for batch in range(int(dataset.train.num_examples / batch_size)):
            for batch in range(int(dataset.num_examples / batch_size)):

                #summary_op = tf.summary.merge(lstm_ctc.summaries)
                summary_op = tf.summary.merge_all()

                #train_x, train_y_sparse, train_sequence_length = dataset.train.next_batch(batch_size)
                train_x, train_y_sparse, train_sequence_length = dataset.next_batch(batch_size)

                feed = {
                    lstm_ctc.input_placeholder : train_x,
                    lstm_ctc.label_sparse_placeholder : train_y_sparse,
                    lstm_ctc.input_seq_len_placeholder : train_sequence_length,
                }

                batch_cost, _, summary = session.run([loss_operation, optimizer_operation, summary_op], feed)
                #batch_cost, _, _, summary = session.run([cost, optimizer, state, summary_op], feed)

                epoch_loss += batch_cost * batch_size

                writer.add_summary(summary, epoch * batch_size + batch)

                ler_loss += session.run(ler_operation, feed) * batch_size

                # Decoding
                d = session.run(decode_operation, feed_dict=feed)
                str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
                # Replacing blank label to none
                str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
                # Replacing space label to space
                str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

                print('Decoded: %s' % str_decoded, )

            #epoch_loss /= dataset.train.num_examples
            #ler_loss /= dataset.train.num_examples
            epoch_loss /= dataset.num_examples
            ler_loss /= dataset.num_examples

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
                  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

            print(log.format(epoch + 1, num_epoches, epoch_loss, ler_loss,
                             0, 0, time.time() - start))


def main(config_path=None, dataset_path=None):

    if config_path is None:
        print("Processing default config.")

        config_path = './src/config/lstm_ctc_VCTK.yml'
        config_reader = ConfigReader(config_path)
    else:
        print("Processing CONFIG in filename: ", config_path)
        config_reader = ConfigReader(config_path)


    if dataset_path is not None:
        print("Setting default train dataset path.")

        train_home_dictionary = dataset_path
    else:
        train_home_dictionary = config_reader.get_train_directory_path()

        print("Setting train dataset path: ", train_home_dictionary)


    if config_reader.get_corpus_name() == 'VCTK':
        print("Integrating VCTK corpus.")

        dataset = VCTKDataset(train_home_dictionary, config_reader.get_num_features(), config_reader.get_num_context())
    else:
        print("Integrating digit corpus.")

        dataset = DataSet.read_number_data_sets(train_home_dictionary)


    train_network(dataset, config_reader)



if __name__ == "__main__":

    import os

    print(os.environ['PYTHONPATH'])

    import sys

    print(sys.path)

    args = sys.argv
    if len(args) == 2:
        main(config_path=args[1])
    elif len(args) == 3:
        main(config_path=args[1], dataset_path=args[2])
    else:
        main()


