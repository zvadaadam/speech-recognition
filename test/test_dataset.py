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



if __name__ == '__main__':

    test_read_digit_dataset('/fixtures/config/lstm_ctc.yml')
    test_read_vctk_dataset('/fixtures/config/lstm_ctc_vctk.yml')