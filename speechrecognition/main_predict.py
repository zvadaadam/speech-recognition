import tensorflow as tf
from speechrecognition.dataset.dataset import Dataset
from speechrecognition.model.rnn_model import RNNModel
from speechrecognition.trainer.trainer import SpeechTrainer


def main_predict(config, x):
    """
    Main function for running prediction process.
    The model is specified in config.
    :param ConfigReader config: config reader object
    :param x: wav file
    """

    # load the wav file
    # init model
    # run prediction