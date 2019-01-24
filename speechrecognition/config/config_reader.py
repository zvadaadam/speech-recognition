import yaml

from speechrecognition.helper.singleton import Singleton

class ConfigReader(object, metaclass=Singleton):
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
            self.dataset = self.dataset['dataset']
            self.features = self.config['feature']
            self.hyperparameters = self.config['hyperparameter']
            self.paths = self.config['path']

    # -----HEADER-----

    @property
    def model_name(self):
        return self.model_name

    # -----DATASET-----

    @property
    def dataset_name(self):
        return self.dataset['name']

    @property
    def dataset_label_type(self):
        return self.dataset['label_type']

    # -----FEATURES-----

    @property
    def feature_size(self):
        return self.features['feature_size']

    @property
    def num_context(self):
        return self.features['num_context']

    # -----HYPERPARAMATERS-----

    @property
    def num_classes(self):
        return self.hyperparameters['num_classes']

    @property
    def num_hidden(self):
        return self.hyperparameters['num_hidden']

    @property
    def num_layers(self):
        return self.hyperparameters['num_layers']

    @property
    def batch_size(self):
        return self.hyperparameters['batch_size']

    @property
    def num_epoches(self):
        return self.hyperparameters['num_epoches']

    @property
    def learning_rate(self):
        return self.hyperparameters['learning_rate']

    @property
    def dropout_prob(self):
        return self.hyperparameters['dropout_prob']

    # -----MODEL-----

    @property
    def get_tensorboard_logs_path(self):
        return self.paths['tensorboard_logs']

    @property
    def get_train_directory_path(self):
        return self.paths['train_directory']

    def get_trained_model_path(self):
        return self.paths['trained_model']
