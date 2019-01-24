import tensorflow as tf

class BaseTrain(object):

    def __init__(self, session, model, dataset, config):
        self.session = session
        self.model = model
        self.dataset = dataset
        self.config = config

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            # run epoch training
            self.train_epoch()
            # increase epoche counter
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
