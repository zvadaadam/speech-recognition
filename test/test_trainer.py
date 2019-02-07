import os

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

def test_training_return_code(trainer):

    print('TEST EXIT CODE')

    assert trainer.exit_code == 0


def test_trained_models_saved(trainer):

    print('TEST TRAINED MODEL SAVED')

    model_path = ABS_PATH + '/fixtures/model/digits/'
    # TODO: assert against a file generated name
    assert os.path.exists(model_path + 'RNN_l2_h100_b16-6150.meta') is True
    assert os.path.exists(model_path + 'RNN_l2_h100_b16-6150.index') is True


def test_tensor_board_logs(trainer):

    print('TEST TENSORBOARD LOGS')

    model_path = ABS_PATH + '/fixtures/model/digits/'
    assert os.path.exists(model_path + '/test') is True
    assert os.path.exists(model_path + '/train') is True
