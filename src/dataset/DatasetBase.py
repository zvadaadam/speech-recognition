import numpy as np
from tqdm import tqdm

from src.utils import audio_utils, text_utils

class DatasetBase(object):

    def __init__(self, num_features, num_context):
        self._audio_filenames = []
        self._label_filenames = []
        self._num_examples = 0

        self.num_features = num_features
        self.num_context = num_context

        self._index_in_epoch = 0
        self._epochs_completed = 0


    @property
    def audio_filenames(self):
        return self._audio_filenames

    @property
    def label_filenames(self):
        return self._label_filenames

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        """" Returns from Datset batch of audios for training/testing of batch_size """

        if batch_size > self._num_examples:
            raise ValueError('Batch size cannot be greather then number of examples in dataset')

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        audio_filenames_batch = []
        label_filenames_batch = []

        if self._index_in_epoch > self._num_examples:
            # count finished epoches
            self._epochs_completed += 1

            # preparse the shuffle of filename data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)

            # shuffle stored filename data
            self._audio_filenames = self._audio_filenames[perm]
            self._label_filenames = self._label_filenames[perm]

            # start next epoch
            start = 0

            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        audios = []
        labels = []

        #print("Preprocessing audio files for batch of size", batch_size)

        #for audio_filename, label_filename in tqdm(zip(self._audio_filenames[start:end], self._label_filenames[start:end]), total=(end - start)):
        for audio_filename, label_filename in zip(self._audio_filenames[start:end], self._label_filenames[start:end]):
            audio_features = audio_utils.audiofile_to_input_vector(audio_filename, self.num_features, self.num_context)

            text_target = text_utils.get_refactored_transcript(label_filename, is_filename=True, is_digit=False)

            audios.append(audio_features)
            labels.append(text_target)

        output_target = np.asarray(labels)
        sparse_targets = text_utils.sparse_tuple_from(output_target)

        # pad audio batch
        train_input, train_length = audio_utils.pad_sequences(audios)

        return train_input, sparse_targets, train_length


