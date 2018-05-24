import tensorflow as tf
import numpy as np

from src.model.LSTMCTC import LSTMCTC
from src.config.ConfigReader import ConfigReader
import src.utils.text_utils as text_utils
import src.utils.audio_utils as audio_utils


def preprocess_speech(wav_file_path):

    audio_features = audio_utils.audiofile_to_input_vector(wav_file_path, 13, 4)

    array = np.expand_dims(audio_features, axis=0)

    return audio_utils.pad_sequences(array)


def predict(wav_file_path, config_file):

    #config_path = './config/lstm_ctc.yml'
    config_reader = ConfigReader(config_file)

    logs_path = config_reader.get_tensorboard_logs_path()

    print(config_reader.get_train_directory_path())

    # Get Network parameters
    num_classes = config_reader.get_num_classes()
    num_epoches = config_reader.get_num_epoches()
    num_hidden = config_reader.get_num_hidden()
    num_layers = config_reader.get_num_layers()
    batch_size = config_reader.get_batch_size()
    num_features = config_reader.get_num_features()
    num_context = config_reader.get_num_context()
    dropout_hidden = config_reader.get_dropout_hidden()
    trained_model = config_reader.get_trained_model_path()

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


    graph = tf.get_default_graph()

    saver = tf.train.Saver()

    with tf.Session() as session:

        saver.restore(session, trained_model)

        input, length = preprocess_speech(wav_file_path)

        decoded_outputs = session.run(decode_operation, feed_dict={lstm_ctc.input_placeholder : input, lstm_ctc.input_seq_len_placeholder : length, lstm_ctc.dropout_placeholder : 1})

        decoded_text = text_utils.index_to_text(decoded_outputs[1])
        print(decoded_text)



if __name__ == "__main__":

    #wav_file_path = '/Users/adamzvada/Desktop/VCTK-Corpus/wav48/p298/p298_003.wav'
    wav_file_path = '../audio_numbers/0/0_jackson_1.wav'

    decoded_text = predict(wav_file_path, '../src/config/lstm_ctc.yml')