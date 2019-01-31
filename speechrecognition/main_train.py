import tensorflow as tf
from speechrecognition.dataset.dataset import Dataset
from speechrecognition.model.rnn_model import RNNModel
from speechrecognition.trainer.trainer import SpeechTrainer


def main_train(config):
    """
    Man for running the training process specified by config
    :param ConfigReader config: config reader object
    """

    dataset = Dataset(config)
    session = tf.Session()

    # TODO: init the right model from config
    model = RNNModel(config)
    # model = BRNNModel(config)

    trainer = SpeechTrainer(session, model, dataset, config)

    trainer.train()


if __name__ == '__main__':

    from speechrecognition.config.config_reader import ConfigReader

    config_path = '/Users/adamzvada/Documents/School/BP/SpeechRecognition/config/lstm_ctc.yml'

    main_train(ConfigReader(config_path))
