import os
import tensorflow as tf
from tqdm import trange
from speechrecognition.helper.tensor_logger import TensorLogger
from speechrecognition.trainer.tensor_iterator import TensorIterator

class BaseTrain(object):

    def __init__(self, session, model, dataset, config):
        self.session = session
        self.model = model
        self.dataset = dataset
        self.config = config

        self.iterator = TensorIterator(dataset, model, session, config)

    def train(self):

        #model_train_inputs = self.prepare_dataset()
        model_train_inputs, train_handle = self.iterator.create_dataset_iterator(mode='train')
        _, test_handle = self.iterator.create_dataset_iterator(mode='test')

        self.train_handle = train_handle
        self.test_handle = test_handle

        self.model.build_model(model_train_inputs)

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(self.init)

        logger = TensorLogger(log_path=self.config.get_tensorboard_logs_path(), session=self.session)

        # tqdm progress bar looping through all epoches
        t_epoches = trange(self.model.cur_epoch_tensor.eval(self.session), self.config.num_epoches() + 1, 1,
                           desc=f'Training {self.config.model_name()}')
        for cur_epoch in t_epoches:
            # run epoch training
            decoded, mean_loss, mean_error = self.train_epoch()

            # Log the loss in the tqdm progress bar
            t_epoches.set_postfix(
                decoded=f'{decoded}',
                epoch_mean_loss='{:05.3f}'.format(mean_loss),
                epoch_mean_error='{:05.3f}'.format(mean_error)
            )

            # log scalars to tensorboard
            summaries_dict = {
                'mean_loss': mean_loss,
                'mean_error': mean_error,
            }
            logger.log_scalars(cur_epoch, summaries_dict=summaries_dict)

            # increase epoche counter
            self.session.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def prepare_dataset(self):
        raise NotImplementedError
