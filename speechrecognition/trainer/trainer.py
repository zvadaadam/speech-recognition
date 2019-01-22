from speechrecognition.base.base_train import BaseTrain


class SpeechTrainer(BaseTrain):

    def __init__(self, config_path):
        super(SpeechTrainer, self).__init__(config_path)

