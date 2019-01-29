from speechrecognition.dataset.digit_dataset import DigitDataset
from speechrecognition.dataset.vctk_dataset import VCTKDataset

class Dataset(object):

    def __init__(self, config):

        self.config = config

        self.init_dataset_engine(self.config)

    def init_dataset_engine(self, config):

        name = config.dataset_name()

        # TODO: get dataset names from static config
        if name == 'digits':
            self.dataset_engine = DigitDataset(
                dataset_path=self.config.dataset_path(),
                num_features=self.config.feature_size(), num_context=self.config.num_context())
        elif name == 'VCTK':
            self.dataset_engine = VCTKDataset(
                dataset_path=self.config.dataset_path(), num_speakers=self.config.num_speakers(),
                num_features=self.config.feature_size(), num_context=self.config.num_context()
            )
        else:
            # TODO: Create my own exepction
            raise Exception('Missing datset engine name.')

