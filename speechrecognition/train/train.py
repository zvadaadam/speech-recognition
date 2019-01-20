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

from src.utils import text_utils


FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

def train_network(dataset, config_reader, is_vctk):

    logs_path = config_reader.get_tensorboard_logs_path()

    # Get Network parameters
    num_classes = config_reader.get_num_classes()
    num_epoches = config_reader.get_num_epoches()
    num_hidden = config_reader.get_num_hidden()
    num_layers = config_reader.get_num_layers()
    batch_size = config_reader.get_batch_size()
    num_features = config_reader.get_num_features()
    num_context = config_reader.get_num_context()
    dropout_hidden = config_reader.get_dropout_hidden()


    graph = tf.Graph()

    lstm_ctc = LSTMCTC(num_hidden, num_layers, num_classes, num_features)
    lstm_ctc.define()
    loss_operation = lstm_ctc.loss_funtion()
    optimizer_operation = lstm_ctc.train_optimizer()
    decode_operation = lstm_ctc.decoder()
    ler_operation = lstm_ctc.compute_label_error_rate(decode_operation)

    # set TF logging verbosity
    tf.logging.set_verbosity(tf.logging.INFO)

    # saving the trainnned model
    saver = tf.train.Saver()

    if is_vctk:
        num_examples = dataset.num_examples
    else:
        num_examples = dataset.train.num_examples

    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        #saver.restore(session, './trained_models/three_speaker_model-102')

        writer = tf.summary.FileWriter(logs_path, graph=session.graph)

        for epoch in range(num_epoches):

            epoch_loss = 0
            ler_loss = 0

            start = time.time()

            current_state = np.zeros((num_layers, 2, batch_size, num_hidden))

            for batch in range(int(num_examples / batch_size)):
            #for batch in range(int(dataset.train.num_examples / batch_size)):
            #for batch in range(int(dataset.num_examples / batch_size)):

                #summary_op = tf.summary.merge(lstm_ctc.summaries)
                summary_op = tf.summary.merge_all()

                if is_vctk:
                    train_x, train_y_sparse, train_sequence_length = dataset.next_batch(batch_size)
                else:
                    train_x, train_y_sparse, train_sequence_length = dataset.train.next_batch(batch_size)


                feed = {
                    lstm_ctc.input_placeholder : train_x,
                    lstm_ctc.label_sparse_placeholder : train_y_sparse,
                    lstm_ctc.input_seq_len_placeholder : train_sequence_length,
                    lstm_ctc.dropout_placeholder : dropout_hidden
                }

                batch_cost, _, summary = session.run([loss_operation, optimizer_operation, summary_op], feed)

                epoch_loss += batch_cost * batch_size

                writer.add_summary(summary, epoch * batch_size + batch)


                #train_x, train_y_sparse, train_sequence_length = dataset.test.next_batch(batch_size)

                feed = {
                    lstm_ctc.input_placeholder: train_x,
                    lstm_ctc.label_sparse_placeholder: train_y_sparse,
                    lstm_ctc.input_seq_len_placeholder: train_sequence_length,
                    lstm_ctc.dropout_placeholder: dropout_hidden
                }


                ler_loss += session.run(ler_operation, feed) * batch_size

                # Decoding
                decoded_str = decode_transcript(session, decode_operation, feed)

                print('Decoded: %s' % decoded_str)

            epoch_loss /= num_examples
            ler_loss /= num_examples


            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"

            print(log.format(epoch + 1, num_epoches, epoch_loss, ler_loss, time.time() - start))

            # Create a checkpoint in every iteration
            if is_vctk:
                saver.save(session, './trained_models/vctk_model', global_step=epoch)
            else:
                saver.save(session, './trained_models/digit_model', global_step=epoch)

        # save a final checkpoint of our model
        if is_vctk:
            saver.save(session, './trained_models/vctk_model_final', global_step=epoch)
        else:
            saver.save(session, './trained_models/digit_model_final', global_step=epoch)


def decode_transcript(session, decode_operation, feed):

    decoded_outputs = session.run(decode_operation, feed_dict=feed)

    decoded_str = text_utils.index_to_text(decoded_outputs[1])

    return decoded_str






def main(config_path=None, dataset_path=None, is_vctk=None):

    if config_path is None:
        print("Processing default config.")

        #config_path = './src/config/lstm_ctc_VCTK.yml'
        config_path = './src/config/lstm_ctc.yml'
        config_reader = ConfigReader(config_path)
    else:
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
        is_vctk = True
    else:
        print("Integrating digit corpus.")
        dataset = DataSet.read_number_data_sets(train_home_dictionary)
        is_vctk = False


    train_network(dataset, config_reader, is_vctk)




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

