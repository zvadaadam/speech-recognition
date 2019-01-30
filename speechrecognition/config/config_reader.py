import yaml
from speechrecognition.helper.singleton import Singleton


## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class ConfigReader(object): #, metaclass=Singleton):
    """" ConfigReader - parse yml config file
    Args:

    Methods:
        get_model_name -> string
    """
    def __init__(self, config_path):

        print("Processing CONFIG in filename: ", config_path)

        yaml.add_constructor('!join', join)

        with open(config_path, 'r') as f:
            config = yaml.load(f)
            self.model_info = config['model_info']
            self.dataset = config['dataset']
            self.features = config['feature']
            self.hyperparameters = config['hyperparameter']
            self.model = config['model']

    # -----HEADER-----

    def model_name(self):
        return self.model_info['name']

    # -----DATASET-----

    def dataset_name(self):
        return self.dataset['name']

    def dataset_path(self):
        return self.dataset['dataset_path']

    def dataset_label_type(self):
        return self.dataset['label_type']

    def num_speakers(self):
        return self.dataset['num_speakers']

    # -----FEATURES-----

    def feature_size(self):
        return self.features['feature_size']

    def num_context(self):
        return self.features['num_context']

    # -----HYPERPARAMATERS-----

    def num_classes(self):
        return self.hyperparameters['num_classes']

    def num_hidden(self):
        return self.hyperparameters['num_hidden']

    def num_layers(self):
        return self.hyperparameters['num_layers']

    def batch_size(self):
        return self.hyperparameters['batch_size']

    def num_epoches(self):
        return self.hyperparameters['num_epoches']

    def num_iterations(self):
        return self.hyperparameters['num_iterations']

    def learning_rate(self):
        return self.hyperparameters['learning_rate']

    def dropout_prob(self):
        return self.hyperparameters['dropout_prob']

    # -----MODEL-----

    def get_tensorboard_logs_path(self):
        return self.model['tensorboard_path']

    def get_trained_model_path(self):
        return self.model['trained_path']

    def model_description(self):
        return self.model['model_description']

    def restore_trained_model(self):
        return self.model['restore_trained_model']