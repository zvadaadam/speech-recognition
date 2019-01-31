# Speech Recognition 🗣📝

End to End __Speech Recognition__ implemented with deep learning framework __Tensorflow__.
Build upon __Recurrent Neural Networks__ with __LSTM__ and __CTC__(Connectionist Temporal Classification).

## 🔨 Install

After cloning the repository, you need to install all the project dependencies etc..

```
$ python setup.py install
```

## 🏄‍ Run

Run it via command line where you can choose to either training or prediction phase.

#### Training 💪

The command for running the training phase.
```
$ python -m speechrecognition train -config ./config/lstm_ctc.yml
```
You need to provide a [configuration file](https://github.com/zvadaadam/speech-recognition/blob/master/config/lstm_ctc.yml) of the training.

#### Prediction 🤔

The command for running the prediction phase.

```
$ python -m speechrecognition predict -audio {path/to/audio-file} -config ./config/lstm_ctc.yml
```
The same configuration file you provided in training phase will be also applied in prediction phase (sucks, i know).
Most importantly, you provide the path to the audio file in wav format, which will be transcribed to text.

## Configuration File

The [configuration file](https://github.com/zvadaadam/speech-recognition/blob/master/config/lstm_ctc.yml)
let's you defined properties and it sets the file paths to datset, training model and tensorboard logs.

The file is in the `yaml` format and this is the predefined structure.

| Section        | Key                   |  Modify |
|----------------|-----------------------|---|
| dataset        | name                  |❗️|
|                | label_type            |  |
|                | lang                  |  |
|                | dataset_path          |❗️|
| feature        | name                  |  |
|                | feature_size          | ️|
| hyperparameter | num_classes           |  |
|                | num_hidden            |  |
|                | num_layers            |  |
|                | batch_size            |  |
|                | num_epoches           |  |
|                | num_iterations        |  |
|                | dropout_prob          |  |
| model          | model_type            |  |
|                | tensorboard_path      |❗️|
|                | trained_path          |❗️|
|                | model_description     |  |
|                | restore_trained_model |  |

## Dataset

It's currently supporting two speech datasets.
* [FreeSpokenDigits](https://github.com/Jakobovski/free-spoken-digit-dataset) (1GB)
* [VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (15GB)

In order to train the model, you need to download your own dataset and store locally and
change the paths to the dataset in the configuration file.

## The Learning Model

### Preprocessing

MFCC

### Model

RNN/BRNN -> Dense Layer -> CTC


### Tensorboard

In the configuration file is defined the path to the Tensorboard logs.
By running this command on the directory, you may see the process of the training phase.
```
$ tensorboard --logdir {path/to/tensorboard-logs}
```

__MI-PYT TODO__:
- [x] Code Refactor
- [x] (TF dataset pipepline - GPU training speed up)
- [x] (Better Tensorboard monitoring)
- [x] (Divide to Train/Test set)
- [ ] (Better Speech Evaluation)
- [ ] Improved sound preprocessing and feature extraction
- [x] (Training model based with bidirectional RNNs)
- [ ] Training model based on Attention Mechanism
- [ ] Training model based on Neural Turing Machine
- [ ] Automated generation of datasets from audiobooks
- [x] Documentation
- [ ] Tests
