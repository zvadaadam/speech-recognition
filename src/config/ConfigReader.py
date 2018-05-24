import yaml


class ConfigReader(object):
    """" ConfigReader - parse yml config file
    Args:

    Methods:
        get_model_name -> string
    """
    def __init__(self, config_path):

        print("Processing CONFIG in filename: ", config_path)

        with open(config_path, 'r') as f:
            self.config = yaml.load(f)
            self.model_name = self.config['model_name']
            self.corpus = self.config['corpus']
            self.features = self.config['feature']
            self.hyperparameters = self.config['hyperparameter']
            self.paths = self.config['path']

    # -----HEADER-----

    def get_model_name(self):
        return self.model_name

    # -----CORPUS-----

    def get_corpus_name(self):
        return self.corpus['name']

    def get_corpus_label_type(self):
        return self.corpus['label_type']

    # -----FEATURES-----

    def get_num_features(self):
        return self.features['num_features']

    def get_num_context(self):
        return self.features['num_context']

    # -----HYPERPARAMATERS-----

    def get_num_classes(self):
        return self.hyperparameters['num_classes']

    def get_num_hidden(self):
        return self.hyperparameters['num_hidden']

    def get_num_layers(self):
        return self.hyperparameters['num_layers']

    def get_batch_size(self):
        return self.hyperparameters['batch_size']

    def get_num_epoches(self):
        return self.hyperparameters['num_epoches']

    def get_dropout_hidden(self):
        return self.hyperparameters['dropout_hidden']

    # -----PATH-----

    def get_tensorboard_logs_path(self):
        return self.paths['tensorboard_logs']

    def get_train_directory_path(self):
        return self.paths['train_directory']

    def get_trained_model_path(self):
        return self.paths['trained_model']
