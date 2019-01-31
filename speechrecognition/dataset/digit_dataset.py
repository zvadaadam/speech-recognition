import os
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split
from speechrecognition.dataset.dataset_base import DatasetBase
from speechrecognition.utils import audio_utils, text_utils


class DigitDataset(DatasetBase):
    """
    Dataset engine/parser for Digit dataset extended from the DatasetBase object.
    Digit dataset from download: https://github.com/Jakobovski/free-spoken-digit-dataset
    """

    def __init__(self, dataset_path, num_features, num_context):
        """
        Initializer of DigitDataset object
        :param str dataset_path: path to digit dataset locally
        :param int num_features: size of feature vector
        :param num_context: NOT USED...
        """
        DatasetBase.__init__(self, num_features, num_context)

        self.dataset_path = dataset_path

        self._train_audios = []
        self._train_labels = []
        self._test_audios = []
        self._test_labels = []

        self.read_digit_dataset(dataset_path)


    def read_digit_dataset(self, dataset_path):
        """
        Function fetches all filenames in dataset folder and loads the file to memory and performs speech preprocessing.
        :param str dataset_path: path to digit dataset locally
        """

        print(f'Preparing Digit Dataset from path {dataset_path}')

        audios = []
        labels = []

        t_digits = trange(10, desc='Preprocessing Digit Dataset')
        for i in t_digits:
            dir = os.path.join(dataset_path, str(i))

            filenames = os.listdir(dir)

            for filename in filenames:
                if 'wav' in filename:
                    wav_path = os.path.join(dir, filename)

                    t_digits.set_postfix(file=f'{wav_path}')

                    audio_features = audio_utils.audiofile_to_input_vector(wav_path, 13, 4)

                    text_target = text_utils.get_refactored_transcript(i, is_filename=False)

                    audios.append(audio_features)
                    labels.append(text_target)


        print(f'Loaded {len(audios)} digit records.')

        audios = np.asarray(audios)
        labels = np.asarray(labels)

        # preshuffle dataset (the main shuffle will be performed in tf.dataset)
        self.shuffle(audios, labels, seed=42)

        # split dataset to train and test
        train_x, test_x, train_y, test_y = train_test_split(audios, labels, test_size=0.3, random_state=42)
        self._train_audios = train_x
        self._train_labels = train_y
        self._test_audios = test_x
        self._test_labels = test_y

        print(f'Divided dataset to {len(self._train_audios)} of training data and {len(self._test_audios)} of testing data.')


def test_dataset():

    dataset_path = '/Users/adamzvada/Documents/School/BP/SpeechRecognition/audio_numbers'

    digit_dataset = DigitDataset(dataset_path=dataset_path, num_features=13, num_context=4)

    train_input, sparse_targets, train_length = digit_dataset.next_batch(8)

    print(train_input)
    print(sparse_targets)
    print(train_length)

if __name__ == "__main__":

    test_dataset()