import tensorflow as tf

class BaseModel(object):
    """
    Base class for Tensorflow model
    """

    def __init__(self, config):
        """
        Initializer of BaseModel object
        :param ConfigReadef config: config reader object
        """
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()


    def save(self, session, global_step=None, write_meta_graph=True):
        """
        Saves the current trained model

        :param tf.session session: tensorflow session
        :param tf.Variable(int) global_step: global step of the training
        :param bool write_meta_graph: flag for whether to save the computational graph model
        """
        save_path = self.config.get_trained_model_path() + self.config.model_description()

        self.saver.save(session, save_path, global_step=global_step or self.global_step_tensor,
                        write_meta_graph=write_meta_graph)

    def load(self, session, model_path=None):
        """

        :param tf.session session: tensorflow session
        :param model_path:
        :return:
        """

        if model_path != None:
            self.saver.restore(session, model_path)

        latest_checkpoint = tf.train.latest_checkpoint(self.config.get_trained_model_path())
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(session, latest_checkpoint)
            print("Model loaded")

    def init_cur_epoch(self):
        """
        Initialize a Tensorflow variable to use it as epoch counter
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        """
        Initialize a tensorflow variable to use it as global step counter
        """
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_saver(self, max_to_keep=None):
        """
        Initialize a tensorflow saver for saving trained models
        :param int max_to_keep: max number of kept checkpoints for particular model
        """
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def build_model(self):
        """
        Function to be overridden in child class which build the whole tensorflow computational graph (model)
        """
        raise NotImplementedError