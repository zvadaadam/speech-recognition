import tensorflow as tf

class TensorIterator(object):


    def __init__(self, dataset, model, session, config):
        self.dataset = dataset
        self.model = model
        self.config = config
        self.session = session

        self.handle_placeholder = tf.placeholder(tf.string, shape=[])

    def create_dataset_iterator(self, mode='train'):
        """
        Create feedable Tensorflow Iterator from dataset

        :param mode:
        :return:
        """

        if mode == 'train':
            x, y_sparse, x_seq_length = self.dataset.dataset_engine.train_dataset()
        else:
            x, y_sparse, x_seq_length = self.dataset.dataset_engine.test_dataset()

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.model.input_placeholder,
             self.model.label_sparse_placeholder,
             self.model.input_seq_len_placeholder)
        )

        buffer_size = len(x) if mode == 'train' else 1

        # TODO: find better buffer_size, know it loads whole dataste in memory?
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(1234)) \
            .batch(self.config.batch_size()) \
            .repeat()

        dataset_iterator = dataset.make_initializable_iterator()

        generic_iterator = tf.data.Iterator.from_string_handle(
            self.handle_placeholder, dataset.output_types, dataset.output_shapes, dataset.output_classes)

        # handle chooses the appropriate iterator
        dataset_handle = self.session.run(dataset_iterator.string_handle())

        # iterators output which needs to be fed to the model
        input, sparse_label, seq_length = generic_iterator.get_next()
        inputs = {
            'input': input,
            'sparse_label': sparse_label,
            'seq_length': seq_length
        }

        # init datset iterator with the data
        feed = {
            self.model.input_placeholder: x,
            self.model.label_sparse_placeholder: y_sparse,
            self.model.input_seq_len_placeholder: x_seq_length,
        }
        self.session.run(dataset_iterator.initializer, feed_dict=feed)

        return inputs, dataset_handle