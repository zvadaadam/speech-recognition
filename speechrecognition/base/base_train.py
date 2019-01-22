import tensorflow as tf

class BaseTrain(object):

    def __init__(self, config_path):
        self.config_path = config_path

    def train(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
