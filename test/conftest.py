import os
import sys
import glob
import pytest
from shutil import copyfile
from click.testing import CliRunner
from speechrecognition import main
from speechrecognition.config.config_reader import ConfigReader
from speechrecognition.dataset.vctk_dataset import VCTKDataset
from speechrecognition.dataset.digit_dataset import DigitDataset


ABS_PATH = os.path.abspath(os.path.dirname(__file__))


# TODO: extend to more config files
@pytest.fixture(params=[
    '/fixtures/config/lstm_ctc.yml'
], scope='session')
def trainer(request):

    def training_cleanup():

        model_path = ABS_PATH + '/fixtures/model/digits/'

        # restore prev checkpoint
        os.remove(model_path + 'checkpoint')
        os.rename(model_path + 'prev_checkpoint', model_path + 'checkpoint')
        # delete new trained models
        for f in glob.glob(model_path + 'RNN_l2_h100_b16-61*'):
            os.remove(f)

    runner = CliRunner()

    copyfile(ABS_PATH + '/fixtures/model/digits/checkpoint', ABS_PATH + '/fixtures/model/digits/prev_checkpoint')

    result = runner.invoke(main.train, [
        '-c', ABS_PATH + request.param
    ])

    yield result

    request.addfinalizer(training_cleanup)


@pytest.fixture(params=[
    ('/fixtures/config/lstm_ctc.yml', '/fixtures/audio_numbers/0/0_jackson_0.wav')
], scope='session')
def predict(request):

    runner = CliRunner()

    result = runner.invoke(main.predict, [
        '-x', ABS_PATH + request.param[1],
        '-c', ABS_PATH + request.param[0]
    ])

    return result


@pytest.fixture(params=[
    '/fixtures/config/lstm_ctc_vctk.yml'
], scope='session')
def vctk_dataset(request):

    config = ConfigReader(ABS_PATH + request.param)

    return VCTKDataset(config.dataset_path(), config.num_speakers(), config.feature_size(), config.num_context())


@pytest.fixture(params=[
    '/fixtures/config/lstm_ctc.yml'
], scope='session')
def digit_dataset(request):

    config = ConfigReader(ABS_PATH + request.param)

    return DigitDataset(config.dataset_path(), config.feature_size(), config.num_context())



