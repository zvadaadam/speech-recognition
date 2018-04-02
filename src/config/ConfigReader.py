import yaml


class ConfigReader(object):
    """" ConfigReader - parse yml config file
    Args:

    Methods:
        get_model_name -> string
    """
    def __init__(self, config_path):

        print("Processing CONFIG in filename: %s", config_path)

        with open(config_path, 'r') as f:
            self.config = yaml.load(f)
            self.model_name = config_path['model_name']
            self.corpus = config_path['corpus']
            self.hyperparameters = config_path['hyperparameter']

    @property
    def get_model_name(self):
        return self.model_name

    @property
    def get_corpus_name(self):
        return self.corpus['name']

    @property
    def get_corpus_label_type(self):
        return self.corpus['label_type']

