import os
import numpy as np
from tqdm import tqdm
from speechrecognition.dataset.dataset_base import DatasetBase
from speechrecognition.utils import audio_utils, text_utils


class DigitDataset(DatasetBase):
    """" VCTKDataset process data from VCTK Corpus
    Args:


    """


    def __init__(self, dataset_path, num_features, num_context):
        DatasetBase.__init__(self, num_features, num_context)

        self.dataset_path = dataset_path

        self.read_digit_dataset(dataset_path)


    def read_digit_dataset(self, dataset_path):

        print(f'Preparing Digit Dataset from path {dataset_path}')

        self._audios = []
        self._labels = []

        for i in tqdm(range(10)):
            dir = os.path.join(dataset_path, str(i))

            filenames = os.listdir(dir)

            for filename in filenames:
                if 'wav' in filename:
                    wav_path = os.path.join(dir, filename)

                    audio_features = audio_utils.audiofile_to_input_vector(wav_path, 13, 4)

                    text_target = text_utils.get_refactored_transcript(i, is_filename=False)

                    self._audios.append(audio_features)
                    self._labels.append(text_target)


        print(f'Loaded {len(self.audios)} digit records.')

        self._audios = np.asarray(self._audios)
        self._labels = np.asarray(self._audios)
        self._num_examples = len(self._audios)



def test_dataset():

    dataset_path = '/Users/adamzvada/Documents/School/BP/SpeechRecognition/audio_numbers'

    digit_dataset = DigitDataset(dataset_path=dataset_path, num_features=13, num_context=4)

    train_input, sparse_targets, train_length = digit_dataset.next_batch(8)

    print(train_input)
    print(sparse_targets)
    print(train_length)

if __name__ == "__main__":

    test_dataset()