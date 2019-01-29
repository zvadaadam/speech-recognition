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
            # self.handle_placeholder: self.train_handle
        })

        return loss, decoded, error

    def prepare_dataset(self):

        x_train, y_train_sparse, train_length = self.dataset.dataset_engine.train_dataset()
        #x_test, y_test_sparse, test_length = self.dataset.dataset_engine.test_dataset()

        self.model.init_placeholders(self.config.feature_size())

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.model.input_placeholder,
             self.model.label_sparse_placeholder,
             self.model.input_seq_len_placeholder)
        )

        # test_dataset = tf.data.Dataset.from_tensor_slices(
        #     (self.model.input_placeholder,
        #      self.model.label_sparse_placeholder,
        #      self.model.input_seq_len_placeholder)
        # )

        # TODO: find better buffer_size, know it loads whole dataste in memory?
        train_dataset = train_dataset.shuffle(buffer_size=len(x_train), seed=tf.set_random_seed(1234))
        train_dataset = train_dataset.batch(self.config.batch_size())
        train_dataset = train_dataset.repeat()

        # test_dataset = test_dataset.shuffle(buffer_size=len(x_test), seed=tf.set_random_seed(1234))
        # test_dataset = test_dataset.batch(self.config.batch_size())
        # test_dataset = test_dataset.repeat()

        train_iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
                                                   output_shapes=train_dataset.output_shapes,
                                                   output_classes=train_dataset.output_classes)

        # test_iterator = tf.data.Iterator.from_structure(output_types=test_dataset.output_types,
        #                                                  output_shapes=test_dataset.output_shapes,
        #                                                  output_classes=test_dataset.output_classes)

        train_init_op = train_iterator.make_initializer(train_dataset)
        # test_init_op = test_iterator.make_initializer(test_dataset)

        # self.handle_placeholder = tf.placeholder(tf.string, shape=[])
        # iter = tf.data.Iterator.from_string_handle(
        #     self.handle_placeholder, train_dataset.output_types, train_dataset.output_shapes)
        # iter.get_next() #???
        input, sparse_label, seq_length = train_iterator.get_next()
        inputs = {
            'input': input,
            'sparse_label': sparse_label,
            'seq_length': seq_length
        }

        self.train_handle = self.session.run(train_iterator.string_handle())
        # self.test_handle = self.session.run(test_iterator.string_handle())

        feed = {
            self.model.input_placeholder: x_train,
            self.model.label_sparse_placeholder: y_train_sparse,
            self.model.input_seq_len_placeholder: train_length,
        }
        self.session.run(train_init_op, feed_dict=feed)

        # feed = {
        #     self.model.input_placeholder: x_test,
        #     self.model.label_sparse_placeholder: y_test_sparse,
        #     self.model.input_seq_len_placeholder: test_length,
        # }
        # self.session.run(test_init_op, feed_dict=feed)

        return inputs

    def decode_transcript(self, decode_sparse):

        decoded_str = text_utils.index_to_text(decode_sparse[1])

        return decoded_str