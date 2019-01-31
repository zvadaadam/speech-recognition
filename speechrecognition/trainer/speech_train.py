from speechrecognition.trainer.base_train import BaseTrain
from speechrecognition.utils import text_utils


class SpeechTrainer(BaseTrain):
    """
    SpeechTrainner extanded from BaseTrain defines the training/test of the model
    """

    def __init__(self, session, model, dataset, config):
        """
        Initializer fot SpeechTrainer object

        :param tf.Session session: tensorflow session
        :param BaseModel model: tensorflow model
        :param BaseDataset dataset: dataset object
        :param ConfigReader config: config reader object
        """
        super(SpeechTrainer, self).__init__(session, model, dataset, config)

    def train_epoch(self, cur_epoche):
        """
        Overridden method for training epoche
        :param int cur_epoche: index of current epoch
        :return: decoded sparse, mean loss value and mean error rate on the training epoch
        """
        num_iterations = self.config.num_iterations()

        mean_loss = 0
        mean_error = 0
        for i in range(num_iterations):
            decoded, loss, error = self.train_step()
            mean_loss += loss
            mean_error += error

            if i % 10 == 0:
                step_num = cur_epoche*num_iterations + i
                decoded_str = self.decode_transcript(decoded)
                self.log_progress(input=(decoded_str, loss, error), num_iteration=step_num, mode='train')

        mean_loss /= num_iterations
        mean_error /= num_iterations

        decoded_str = self.decode_transcript(decoded)

        return decoded_str, mean_loss, mean_error

    def train_step(self):
        """
        Overridden for training step, it runs training session in the computational graph and increaet the global step
        :return: decoded sparse, loss value and error rate on the training step
        """

        loss, _, decoded, error = self.session.run([
            self.model.loss, self.model.optimizer, self.model.decoder, self.model.label_error
        ], feed_dict={
            self.model.dropout_placeholder: self.config.dropout_prob(),
            self.iterator.handle_placeholder: self.train_handle
        })

        # increase global step counter
        self.session.run(self.model.increment_global_step_tensor)

        return decoded, loss, error

    def test_step(self):
        """
        Overridden for test step, it runs test session in the computational graph
        :return: decoded sparse, loss value and error rate on the test step
        """
        loss, decoded, error = self.session.run([
            self.model.loss, self.model.decoder, self.model.label_error
        ], feed_dict={
            self.model.dropout_placeholder: 1,
            self.iterator.handle_placeholder: self.test_handle
        })

        decoded_str = self.decode_transcript(decoded)

        return decoded_str, loss, error

    def log_progress(self, input, num_iteration, mode):
        """
        Overridden method to log the training/testing progress in Tensorboard
        :param dict input: inputs to be logged, output of training/testing step
        :param int num_iteration: number of iteration (global step)
        :param str mode: mode [train || test]
        """
        summaries_dict = {
            'text': input[0],
            'loss': input[1],
            'error': input[2],
        }

        self.logger.log_scalars(num_iteration, summarizer=mode, summaries_dict=summaries_dict)

    def update_progress_bar(self, t_bar, train_input, test_input):
        """
        Overridden method to update the tqdm progress bar
        :param tqdm t_bar: tqdm object
        :param dict train_input: inputs to be logged, output of training step
        :param dict test_input: inputs to be logged, output of testing step
        """
        # Log the loss in the tqdm progress bar
        t_bar.set_postfix(
            #decoded=f'{train_input[0]}',
            train_loss='{:05.3f}'.format(train_input[1]),
            train_error='{:05.3f}'.format(train_input[2]),
            test_loss='{:05.3f}'.format(test_input[1]),
            test_error='{:05.3f}'.format(test_input[2]),
        )

    def decode_transcript(self, decode_sparse):
        """
        Decoded the transcript from ascii to string
        :param tuple(sparse) decode_sparse: decoded sparse
        :return: decoded string
        """
        decoded_str = text_utils.index_to_text(decode_sparse[1])

        return decoded_str