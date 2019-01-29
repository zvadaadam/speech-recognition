import os
import tensorflow as tf

class TensorLogger:


    def __init__(self, log_path, session):
        self.session = session
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(log_path, "train"), self.session.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(log_path, "test"))

    # it can summarize scalars
    def log_scalars(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """

        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer

        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():

                    if tag not in self.summary_ops:

                        if isinstance(value, str):
                            self.summary_placeholders[tag] = tf.placeholder(tf.string, shape=(None), name=tag)
                            self.summary_ops[tag] = tf.summary.text(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_placeholders[tag] = tf.placeholder(tf.float32, value.shape, name=tag)
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

                    summary_list.append(self.session.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)

                summary_writer.flush()
