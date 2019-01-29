import tensorflow as tf
from tqdm import trange
from speechrecognition.base.base_train import BaseTrain
from speechrecognition.utils import text_utils


class SpeechTrainer(BaseTrain):

    def __init__(self, session, model, dataset, config):
        super(SpeechTrainer, self).__init__(session, model, dataset, config)

    def train_epoch(self):
        num_iterations = self.config.num_iterations()

        mean_loss = 0
        mean_error = 0
        for i in range(num_iterations):
            loss, decoded, error = self.train_step()
            mean_loss += loss
            mean_error += error

        mean_loss /= num_iterations
        mean_error /= num_iterations

        decoded_str = self.decode_transcript(decoded)

        return decoded_str, mean_loss, mean_error

    def train_step(self):

        loss, _, decoded, error = self.session.run([
            self.model.loss, self.model.optimizer, self.model.decoder, self.model.label_error
        ], feed_dict={
            self.model.dropout_placeholder: self.config.dropout_prob(),
            self.iterator.handle_placeholder: self.train_handle
        })

        loss, _, decoded, error = self.session.run([
            self.model.loss, self.model.optimizer, self.model.decoder, self.model.label_error
        ], feed_dict={
            self.model.dropout_placeholder: self.config.dropout_prob(),
            self.iterator.handle_placeholder: self.test_handle
        })

        return loss, decoded, error


    def decode_transcript(self, decode_sparse):

        decoded_str = text_utils.index_to_text(decode_sparse[1])

        return decoded_str