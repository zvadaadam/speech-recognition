import tensorflow as tf

class BaseModel(object):

    def __init__(self, config_path):
        self.config_path = config_path

    # TODO: save tf checkpoint
    def save(self, session):
        raise NotImplementedError

    # TODO: load tf checkpoint
    def load(self, session):
        raise NotImplementedError