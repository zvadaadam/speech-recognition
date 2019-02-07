import os

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

def test_read_digit_dataset(digit_dataset):

    # TODO: change to read_dataset
    digit_dataset.read_digit_dataset()

    assert len(digit_dataset._train_audios) == 77
    assert len(digit_dataset._test_audios) == 33


def test_read_vctk_filenames(vctk_dataset):

    audio_filenames, label_filenames = vctk_dataset.get_dataset_filenames()

    assert len(audio_filenames) + len(label_filenames) == 20

def test_read_vctk_dataset(vctk_dataset):

    vctk_dataset.read_dataset()

    assert len(vctk_dataset._train_audios) == 7
    assert len(vctk_dataset._test_audios) == 3


def test_speech_targets(vctk_dataset):

    x, y_sparse, x_length = vctk_dataset.transform_to_speech_targets(vctk_dataset._train_audios, vctk_dataset._train_labels)

    assert x.shape[1] > 0
    assert x.shape[2] == vctk_dataset.num_features
    assert x.shape[0] == len(x_length)
    assert y_sparse[0].shape[0] == y_sparse[1].shape[0]



if __name__ == '__main__':

    from speechrecognition.config.config_reader import ConfigReader
    from speechrecognition.dataset.vctk_dataset import VCTKDataset

    config = ConfigReader(ABS_PATH + '/fixtures/config/lstm_ctc_vctk.yml')

    vctk_dataset = VCTKDataset(config.dataset_path(), config.num_speakers(), config.feature_size(), config.num_context())
    test_speech_targets(vctk_dataset)