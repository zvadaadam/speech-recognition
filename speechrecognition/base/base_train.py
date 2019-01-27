import tensorflow as tf

class BaseTrain(object):

    def __init__(self, session, model, dataset, config):
        self.session = session
        self.model = model
        self.dataset = dataset
        self.config = config

    def train(self):

        inputs = self.init_dataset()

        self.model.build_model(inputs)

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(self.init)

        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.session), self.config.num_epoches() + 1, 1):

            print('TRAINING')

            # run epoch training
            self.train_epoch()
            # increase epoche counter
            self.session.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def init_dataset(self):
        raise NotImplementedError
