from tqdm import tqdm
from speechrecognition.base.base_train import BaseTrain
from speechrecognition.utils.text_utils import index_to_text


class SpeechTrainer(BaseTrain):

    def __init__(self, session, model, dataset, config):
        super(SpeechTrainer, self).__init__(session, model, dataset, config)

    def train_epoch(self):
        num_iterations = self.config.num_iterations()

        losses = []
        errors = []
        for i in range(num_iterations):
            loss, decoded, error = self.train_step()
            losses.append(loss)
            errors.append(error)

        # from sparse taking just values to decode
        decoded_chars = index_to_text(decoded[1])
        print(f'Epoch#{i}: {decoded_chars}')

    def train_step(self):

        batch_size = self.config.batch_size()

        batch_x, batch_y, batch_seq_length = self.dataset.dataset_engine.next_batch(batch_size)

        feed = {
            self.model.input_placeholder: batch_x,
            self.model.label_sparse_placeholder: batch_y,
            self.model.input_seq_len_placeholder: batch_seq_length,
            self.model.dropout_placeholder: self.config.dropout_prob(),
        }

        loss, _, decoded, error = self.session.run([
            self.model.loss, self.model.optimizer, self.model.decoder, self.model.label_error
        ], feed)

        return loss, decoded, error
