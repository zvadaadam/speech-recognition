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
            self.model.dropout_placeholder: self.config.dropout_prob()
        })

        return loss, decoded, error

    def prepare_dataset(self):

        # TODO: add the test_dataset!
        train_x, train_sparse_y, train_length = self.dataset.dataset_engine.train_dataset()

        self.model.init_placeholders(self.config.feature_size())

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.model.input_placeholder,
             self.model.label_sparse_placeholder,
             self.model.input_seq_len_placeholder)
        )
        train_dataset = train_dataset.batch(self.config.batch_size())
        train_dataset = train_dataset.repeat()
        # TODO: check if shuffle works
        train_dataset = train_dataset.shuffle(buffer_size=100)

        iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
                                                   output_shapes=train_dataset.output_shapes,
                                                   output_classes=train_dataset.output_classes)

        train_init_op = iterator.make_initializer(train_dataset)

        feed = {
            self.model.input_placeholder: train_x,
            self.model.label_sparse_placeholder: train_sparse_y,
            self.model.input_seq_len_placeholder: train_length,
        }
        self.session.run(train_init_op, feed_dict=feed)

        input, sparse_label, seq_length = iterator.get_next()
        inputs = {
            'input': input,
            'sparse_label': sparse_label,
            'seq_length': seq_length
        }

        return inputs


    def decode_transcript(self, decode_sparse):

        decoded_str = text_utils.index_to_text(decode_sparse[1])

        return decoded_str