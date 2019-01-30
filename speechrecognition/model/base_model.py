import tensorflow as tf

class BaseModel(object):

    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save current checkpoint
    def save(self, session, global_step=None, write_meta_graph=True):
        save_path = self.config.get_trained_model_path() + self.config.model_description()

        self.saver.save(session, save_path, global_step=global_step or self.global_step_tensor,
                        write_meta_graph=write_meta_graph)

    # load latest checkpoint
    def load(self, sess, model_path=None):

        if model_path != None:
            self.saver.restore(sess, model_path)

        latest_checkpoint = tf.train.latest_checkpoint(self.config.get_trained_model_path())
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_saver(self, max_to_keep=None):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def build_model(self):
        raise NotImplementedError