import os
from tqdm import tqdm

from speechrecognition.dataset.dataset_base import DatasetBase
from speechrecognition.utils import audio_utils, text_utils


class VCTKDataset(DatasetBase):
    """" VCTKDataset process data from VCTK Corpus
    Args:
        dataset_path
        num_features
        num_context
    """


    def __init__(self, dataset_path, num_speakers, num_features, num_context):
        DatasetBase.__init__(self, num_features, num_context)
        self.dataset_path = dataset_path

        #self.read_vctk_dataset(num_speakers=3)
        #self.read_and_cache_vctk_dataset(dataset_path, num_speakers)
        self.load_pickle_dataset('test')

        print(self._labels)


    def read_vctk_dataset(self, dataset_path=None, num_speakers=None):
        """" Retrives filenames from VCTK structues corpus and stores them in VCTKDataset object
        Args:
            dataset_path (string) - given path to home directory of dataset
        """

        if dataset_path is not None:
            self.dataset_path = dataset_path

        print("Retriving all filenames for VCTK training dataset from path", self.dataset_path)

        audio_dataset_path = os.path.join(self.dataset_path, 'wav48')
        label_dataset_path = os.path.join(self.dataset_path, 'txt')

        # gets list of directories for diffrent speakers
        speakers_dirs = os.listdir(audio_dataset_path)

        if num_speakers is not None:
            speakers_dirs = speakers_dirs[0:num_speakers]
            print('Number of speakers: ', num_speakers)
        else:
            print('All speakers')

        for speaker_dir in tqdm(speakers_dirs, total=len(speakers_dirs)):

            # get full paths for speakers
            speaker_audio_path = os.path.join(audio_dataset_path, speaker_dir)
            speaker_label_path = os.path.join(label_dataset_path, speaker_dir)

            # ignore inconsistency in data or anything that is not directory
            if not os.path.isdir(speaker_audio_path) or not os.path.isdir(speaker_label_path):
                continue

            # Getting full paths to all audios and labels of the speaker
            audio_paths = [os.path.join(speaker_audio_path, speaker_filename) for speaker_filename in os.listdir(speaker_audio_path)]
            label_paths = [os.path.join(speaker_label_path, speaker_filename) for speaker_filename in os.listdir(speaker_label_path)]

            audio_paths.sort()
            label_paths.sort()

            # concatenate speaker filenames
            self._audio_filenames = self._audio_filenames + audio_paths
            self._label_filenames = self._label_filenames + label_paths

        print()

        self._num_examples = len(self._audio_filenames)

    def read_and_cache_vctk_dataset(self, dataset_path=None, num_speakers=None):

        if dataset_path is not None:
            self.dataset_path = dataset_path

        print("Retriving all filenames for VCTK training dataset from path", self.dataset_path)

        audio_dataset_path = os.path.join(self.dataset_path, 'wav48')
        label_dataset_path = os.path.join(self.dataset_path, 'txt')

        # gets list of directories for diffrent speakers
        speakers_dirs = os.listdir(audio_dataset_path)

        if num_speakers is not None:
            speakers_dirs = speakers_dirs[0:num_speakers]
            print('Number of speakers: ', num_speakers)
        else:
            print('All speakers')

        for speaker_dir in speakers_dirs:

            # get full paths for speakers
            speaker_audio_path = os.path.join(audio_dataset_path, speaker_dir)
            speaker_label_path = os.path.join(label_dataset_path, speaker_dir)

            # ignore inconsistency in data or anything that is not directory
            if not os.path.isdir(speaker_audio_path) or not os.path.isdir(speaker_label_path):
                continue

            # Getting full paths to all audios and labels of the speaker
            audio_paths = [os.path.join(speaker_audio_path, speaker_filename) for speaker_filename in os.listdir(speaker_audio_path)]
            label_paths = [os.path.join(speaker_label_path, speaker_filename) for speaker_filename in os.listdir(speaker_label_path)]

            audio_paths.sort()
            label_paths.sort()

            # concatenate speaker filenames
            self._audio_filenames = self._audio_filenames + audio_paths
            self._label_filenames = self._label_filenames + label_paths


        self._num_examples = len(self._audio_filenames)

        self._audios = []
        self._labels = []

        print('Preprocessing anf Feature extraction...')

        for audio_filename, label_filename in tqdm(zip(self._audio_filenames, self._label_filenames), total=self._num_examples):
            audio_features = audio_utils.audiofile_to_input_vector(audio_filename, self.num_features, self.num_context)

            text_target = text_utils.get_refactored_transcript(label_filename, is_filename=True, is_digit=False)

            self._audios.append(audio_features)
            self._labels.append(text_target)



def test_dataset():

    dataset_path = '/Users/adamzvada/Documents/School/BP/VCTK-Corpus'

    vctk_dataset = VCTKDataset(dataset_path=dataset_path, num_speakers=1, num_features=13, num_context=4)

    #vctk_dataset.save_pickle_dataset('test')
    train_input, sparse_targets, train_length = vctk_dataset.next_batch(8)

    print(train_input)
    print(sparse_targets)
    print(train_length)

if __name__ == "__main__":

    test_dataset()