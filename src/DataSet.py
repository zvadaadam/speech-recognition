"""
    Dataset class parse and stores training and validation data.
    Also handles preprocessing of data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import os
from sklearn.model_selection import train_test_split

from src import audio_utils, text_utils

# Datasets - encapsulate DataSet objects
Datasets = collections.namedtuple("Datasets", ["train", "validation", "test"])

Audio = collections.namedtuple("Audio", ["sample_rate", "data"])

# Object which stores train data
class DataSet(object):


    def __init__(self, audios, labels, num_examples):
        self._audios = np.array(audios)
        self._labels = np.array(labels)
        self._sparse_targets = []
        self._sequence_lengths = []
        self._num_examples = num_examples


        # tracking ?
        self._index_in_epoch = 0

        # tracking ?
        self._epochs_completed = 0


    @property
    def audios(self):
        return self._audios

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


    def next_batch(self, batch_size):
        """" Returns from Datset batch of audios for training/testing of batch_size """

        if batch_size > self._num_examples:
            raise ValueError('Batch size cannot be greather then number of examples in dataset')

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # finished epoches
            self._epochs_completed += 1

            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)

            self._audios = self._audios[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0

            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        output_target = np.asarray(self._labels[start:end])
        sparse_targets = text_utils.sparse_tuple_from(output_target)

        # Pad Audio batch
        train_input, train_length = audio_utils.pad_sequences(self._audios[start:end])

        return train_input, sparse_targets, train_length



def read_number_data_sets(train_data_dir):
    """
        Read data set from given dictionary
    """

    audios = []
    labels = []

    print("Ready to read dataset.")

    for i in range(10):
        dir = os.path.join(train_data_dir, str(i))

        filenames = os.listdir(dir)

        for filename in filenames:
            if 'wav' in filename:
                print("Processing file ", filename)

                wav_path = os.path.join(dir, filename)

                #audio_features = audio_utils.audio_to_feature_vectors(wav_path, 13)
                audio_features = audio_utils.audiofile_to_input_vector(wav_path, 13, 4)
                print(audio_features.shape)
                text_target = text_utils.get_refactored_transcript(i, is_filename=False)

                audios.append(audio_features)
                labels.append(text_target)


        print("FINISHED WITH NUMBER ", i)


    labels = np.asarray(labels)
    audios = np.asarray(audios)

    train_x, test_x, train_y, test_y = train_test_split(audios, labels, test_size=0.1)

    train_dataset = DataSet(train_x, train_y, len(train_x))

    validation_dataset = []

    test_dataset = DataSet(test_x, test_y, len(train_x))

    return Datasets(train=train_dataset, validation=validation_dataset, test=test_dataset)


