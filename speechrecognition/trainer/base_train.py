import os
import tensorflow as tf
from tqdm import trange
from speechrecognition.utils.tensor_logger import TensorLogger
from speechrecognition.trainer.tensor_iterator import TensorIterator

class BaseTrain(object):

    def __init__(self, session, model, dataset, config):
        self.session = session
        self.model = model
        self.dataset = dataset
        self.config = config

        self.iterator = TensorIterator(dataset, model, session, config)
        self.logger = TensorLogger(log_path=self.config.get_tensorboard_logs_path(), session=self.session)


    def train(self):

        model_train_inputs, train_handle = self.iterator.create_dataset_iterator(mode='train')
        _, test_handle = self.iterator.create_dataset_iterator(mode='test')

        self.train_handle = train_handle
        self.test_handle = test_handle

        self.model.build_model(model_train_inputs)

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(self.init)

        self.model.init_saver(max_to_keep=2)

        # restore latest checkpoint model
        self.model.load(self.session, self.config.restore_trained_model())

        # tqdm progress bar looping through all epoches
        t_epoches = trange(self.model.cur_epoch_tensor.eval(self.session), self.config.num_epoches() + 1, 1,
                           desc=f'Training {self.config.model_name()}')
        for cur_epoch in t_epoches:

            # run epoch training
            train_output = self.train_epoch(cur_epoch)
            # run model on test set
            test_output = self.test_step()


            self.log_progress(train_output, num_iteration=cur_epoch * self.config.num_iterations(), mode='train')
            self.log_progress(test_output, num_iteration=cur_epoch * self.config.num_iterations(), mode='test')

            self.update_progress_bar(t_epoches, train_output, test_output)

            # increase epoche counter
            self.session.run(self.model.increment_cur_epoch_tensor)

            self.model.save(self.session, write_meta_graph=False)

        # finale save model - creates checkpoint
        self.model.save(self.session, write_meta_graph=True)


    def train_epoch(self, cur_epoche):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def test_step(self):
        raise NotImplementedError

    def log_progress(self, input, num_iteration, mode):
        raise NotImplementedError

    def update_progress_bar(self, t_bar, input):
        raise NotImplementedError
