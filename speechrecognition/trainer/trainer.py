import tensorflow as tf
from tqdm import tqdm
from speechrecognition.base.base_train import BaseTrain
from speechrecognition.utils import text_utils


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
        decoded_str = self.decode_transcript(decoded)
        print(f'Epoch#{i}: {decoded_str}')

    def train_step(self):

        batch_size = self.config.batch_size()

        # batch_x, batch_y, batch_seq_length = self.dataset.dataset_engine.next_batch(batch_size)

        # feed = {
        #     self.model.input_placeholder: batch_x,
        #     self.model.label_sparse_placeholder: batch_y,
        #     self.model.input_seq_len_placeholder: batch_seq_length,
        #     self.model.dropout_placeholder: self.config.dropout_prob(),
        # }

        # loss, _, decoded, error = self.session.run([
        #     self.model.loss, self.model.optimizer, self.model.decoder, self.model.label_error
        # ], feed)

        loss, _, decoded, error = self.session.run([
            self.model.loss, self.model.optimizer, self.model.decoder, self.model.label_error
        ])

        return loss, decoded, error

    def init_dataset(self):

        # TODO: add test_dataset
        train_x, train_sparse_y, train_length = self.dataset.dataset_engine.train_dataset()

        #sparse_label = tf.SparseTensorValue(indices=train_sparse_y[0], values=train_sparse_y[1], dense_shape=train_sparse_y[2])

        self.model.init_placeholders(self.config.feature_size())

        #train_dataset = tf.data.Dataset.from_tensor_slices((train_x, sparse_label, train_length))
        train_dataset = tf.data.Dataset.from_tensor_slices((self.model.input_placeholder, self.model.label_sparse_placeholder, self.model.input_seq_len_placeholder))
        train_dataset = train_dataset.batch(self.config.batch_size())
        #train_dataset = train_dataset.map(lambda x, y, z: tf.sparse.to_dense(y))
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(buffer_size=100)

        iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types, output_shapes=train_dataset.output_shapes,
                                                   output_classes=train_dataset.output_classes)
        train_init_op = iterator.make_initializer(train_dataset)
        #next_element = iterator.get_next()
        #print(self.session.run(next_element))

        feed = {
            self.model.input_placeholder: train_x,
            self.model.label_sparse_placeholder: train_sparse_y,
            self.model.input_seq_len_placeholder: train_length
            #self.model.dropout_placeholder: self.config.dropout_prob(),
        }

        self.session.run(train_init_op, feed_dict=feed)
        #self.session.run(train_init_op)

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