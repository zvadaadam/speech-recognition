Configuration
=============

Speech Recognition requires configuration file.

Configuration file has to in .yaml format with predefined structure.

Example of such a configuration file.
.. configuration-block::

    .. code-block:: yaml

        model_info:
            name: multilayer_lstm_ctc
        dataset:
            name: &dataset_name digits
            label_type: numbers
            lang: ENG
            dataset_path: /Users/adamzvada/Documents/School/BP/SpeechRecognition/audio_numbers
        feature:
            name: mfcc
            feature_size: 13
            num_context: 4
        hyperparameter:
            num_classes: 28 #ord('z') - ord('a') + 1(space) + 1(blank) + 1
            num_hidden: &num_hidden 100
            num_layers: &num_layers 2
            batch_size: &batch_size 16
            num_epoches: 120
            num_iterations: 50
            optimizer: Adam
            learning_rate: 0.9
            weight_init: 0.1
            clip_grad: 5.0
            clip_activation: 50
            dropout_prob: 0.5
        model:
            model_type: &model_type RNN
            tensorboard_path: /Users/adamzvada/Documents/School/BP/SpeechRecognition/tensorboard_log/digits
            trained_path:  !join [/Users/adamzvada/Documents/School/BP/SpeechRecognition/trained_models/, *dataset_name, /]
            model_description: !join [*model_type, _, l, *num_layers, _, h, *num_hidden, _, b, *batch_size]
            restore_trained_model:
