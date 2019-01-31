import yaml
from speechrecognition.helper.singleton import Singleton


## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class ConfigReader(object): #, metaclass=Singleton):
    """"
    Reader of .yaml config file with getter methods.
    """
    def __init__(self, config_path):
        """
        Initializer for project config

        :param str config_path: path to your .yaml config file
        """

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
        """
        Model info name, just for your purposes.
        """
        return self.model_info['name']

    # -----DATASET-----

    def dataset_name(self):
        """
        Name of your training dataset.
        Using this name, the Dataset class will choose which dataset parser/engine to use.
        """
        return self.dataset['name']

    def dataset_path(self):
        """
        Path to the local dataset, requires the official structure.
        """
        return self.dataset['dataset_path']

    def dataset_label_type(self):
        """
        Indicator how the labels are stored in dataset.
        Not being used so far. Will I ever?
        """
        return self.dataset['label_type']

    def num_speakers(self):
        """
        Number of dataset speakers to be processed.
        Some of the dataset engines are supporting this value (VCTK).
        """
        return self.dataset['num_speakers']

    # -----FEATURES-----

    def feature_size(self):
        """
        Length of the speech feature vector!
        To be more precious, it's number of cepstral coefficients from MFCC speech preprocessing.
        """
        return self.features['feature_size']

    def num_context(self):
        """
        # TODO: connect or change
        Param for MFCC, will be changed...
        """
        return self.features['num_context']

    # -----HYPERPARAMATERS-----

    def num_classes(self):
        """
        Number of output classes from the deep learning model.
        Since we do speech recognition, it's the number of alphabet number plus blank symbol.
        """
        return self.hyperparameters['num_classes']

    def num_hidden(self):
        """
        Number of hidden cells in the layer.
        TODO: we should be able to set different number of hidden cells in each layer, duh!
        """
        return self.hyperparameters['num_hidden']

    def num_layers(self):
        """
        Number of layers for the main of the network.
        """
        return self.hyperparameters['num_layers']

    def batch_size(self):
        """
        Batch size for training step.
        How many training instances will be fed in one training step...
        """
        return self.hyperparameters['batch_size']

    def num_epoches(self):
        """
        Number of training epoches.
        """
        return self.hyperparameters['num_epoches']

    def num_iterations(self):
        """
        Number of training steps(iterations) in each epoch.
        """
        return self.hyperparameters['num_iterations']

    def learning_rate(self):
        """
        Learning rate for the optimizer.
        Not hooked up, yet...
        TODO: hook up learning rate
        """
        return self.hyperparameters['learning_rate']

    def dropout_prob(self):
        """
        Probability of droping the cell.
        That's how we fight with overfitting {emoticon with gangsta glasses :D}
        """
        return self.hyperparameters['dropout_prob']

    # -----MODEL-----

    def get_tensorboard_logs_path(self):
        """
        Path to the Tensorboard logs of the training/testing summary.
        This is gonna be your Tensorboard logdir.
        """
        return self.model['tensorboard_path']

    def get_trained_model_path(self):
        """
        Path to the trained models.
        That's the place where the latest checkpoints will be loaded from or directory where it will be saved.
        """
        return self.model['trained_path']

    def model_description(self):
        """
        Model description name aggregated from the main model params.
        """
        return self.model['model_description']

    def restore_trained_model(self):
        """
        Full path to trained model which you want to load up.
        If you leave it empty, then the latest checkpoint will be processed or maybe nothing :D
        """
        return self.model['restore_trained_model']