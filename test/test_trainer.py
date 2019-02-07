import os
import sys
import glob
import pytest
from shutil import copyfile
from click.testing import CliRunner
from speechrecognition import main

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

# TODO: extend to more config files
@pytest.mark.parametrize(
    ['config'],
    ['/fixtures/config/lstm_ctc.yml'],
)
def test_run_training(config):

    runner = CliRunner()

    copyfile(ABS_PATH + '/fixtures/model/digits/checkpoint', ABS_PATH + '/fixtures/model/digits/prev_checkpoint')

    result = runner.invoke(main.train, [
        '-c', config
    ])

    print(result.output)

    assert result.exit_code == 0

    # TODO: change to generated file name
    model_path = ABS_PATH + '/fixtures/model/digits/'

    # Are trained models saved?
    assert os.path.exists(model_path + 'RNN_l2_h100_b16-6150.meta') is True
    assert os.path.exists(model_path + 'RNN_l2_h100_b16-6150.index') is True

    # restore prev checkpoint
    os.remove(model_path + 'checkpoint')
    os.rename(model_path + 'prev_checkpoint', model_path + 'checkpoint')

    # delete new trained models
    for f in glob.glob(model_path + 'RNN_l2_h100_b16-61*'):
        os.remove(f)

    # Are Tensorboard logs generated?
    assert os.path.exists(model_path + '/test') is True
    assert os.path.exists(model_path + '/train') is True


def test_run_prediction():

    runner = CliRunner()

    ABS_PATH = os.path.abspath(os.path.dirname(__file__))
    print(ABS_PATH)

    result = runner.invoke(main.predict, [
        '-x', '/Users/adamzvada/Documents/School/BP/SpeechRecognition/test/fixtures/audio_numbers/0/0_jackson_0.wav',
        '-c', '/Users/adamzvada/Documents/School/BP/SpeechRecognition/config/lstm_ctc.yml'
    ])

    assert result.exit_code == 0


if __name__ == '__main__':

    test_run_training('/Users/adamzvada/Documents/School/BP/SpeechRecognition/test/fixtures/config/lstm_ctc.yml')