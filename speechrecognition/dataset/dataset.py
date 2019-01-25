from speechrecognition.dataset.DigitDataset import DigitDataset
from speechrecognition.dataset.VCTKDataset import VCTKDataset

class Dataset(object):

    def __init__(self, config):

        self.config = config

        self.init_dataset_engine(self.config)

    def init_dataset_engine(self, config):

        name = config.dataset_name()

        # TODO: get dataset names from static config
        if name == 'digits':
            self.dataset_engine = DigitDataset(
                self.config.dataset_path(), self.config.feature_size(), self.config.num_context())
        elif name == 'VCTK':
            self.dataset_engine = VCTKDataset(
                self.config.dataset_path(), self.config.feature_size(), self.config.num_context()
            )
        else:
            # TODO: Create my own exepction
            raise Exception('Missing datset engine name.')

