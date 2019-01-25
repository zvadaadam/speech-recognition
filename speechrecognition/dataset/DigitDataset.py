from speechrecognition.dataset.dataset_base import DatasetBase

import os

class DigitDataset(DatasetBase):
    """" VCTKDataset process data from VCTK Corpus
    Args:


    """


    def __init__(self, dataset_path, num_features, num_context):
        DatasetBase.__init__(self, num_features, num_context)

        self.dataset_path = dataset_path


    def read_digit_dataset(self, dataset_path):

        for i in range(10):
            dir = os.path.join(dataset_path, str(i))

            filenames = os.listdir(dir)

            for filename in filenames:
                wav_path = os.path.join(dir, filename)
                self.audio_filenames.append(wav_path)

        self.num_examples = self.audio_filenames.count()