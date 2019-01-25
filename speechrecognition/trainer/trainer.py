from tqdm import tqdm
from speechrecognition.base.base_train import BaseTrain
from speechrecognition.utils.text_utils import index_to_text


class SpeechTrainer(BaseTrain):

    def __init__(self, session, model, dataset, config):
        super(SpeechTrainer, self).__init__(session, model, dataset, config)

    def train_epoch(self):
        num_epoche = self.config.num_epoches()

        losses = []
        errors = []
        for i in tqdm(range(num_epoche)):
            loss, decoded, error = self.train_step()
            losses.append(loss)
            errors.append(error)

            decoded_chars = index_to_text(decoded)
            print(f'Epoch#{i}: {decoded_chars}')


    def train_step(self):

        batch_size = self.config.batch_size()

        batch_x, batch_y, batch_seq_length = next(self.dataset.next_batch(batch_size))

        feed = {
            self.model.x: batch_x,
            self.model.y: batch_y,
            self.model.seq_length: batch_seq_length,
            self.model.dropout_prob: self.config.dropout_probability(),
        }

        loss, _, decoded, error = self.session.run([
            self.model.loss, self.model.optimizer, self.model.decoder, self.model.label_error
        ], feed)

        return loss, decoded, error
