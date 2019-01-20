# Speech Recognition

Speech recognition implemented by using __Tensorflow__ for deep learning.
We are using __recurrent neural network__ with __LSTM__ nodes and in order to deal with sequencing problem we apply __CTC__(Connectionist Temporal Classification).

__MI-PYT TODO__:
- [ ] Code Refactor
- [ ] Improved sound preprocessing and feature extraction
- [ ] Training model based on Attention Mechanism
- [ ] Training model based on Neural Turing Machine
- [ ] Automated generation of datasets from audiobooks
- [ ] Documentation
- [ ] Tests


## Training data
In order to train our neural network we have to download training data.
I'm using __VCTK Corpus__ which will do the trick for my purpose. If you are aiming to lower error rates I strongly recommend using more training data.  
⚠️ VCTK Corpus - 15GB
```
# LINUX
wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
# MAC
curl -O http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
```

### Requirements
The speech recognition was developed using Python 3.6.2. and project requirements are located in ```requirements.txt```.
```
numpy==1.13.3
python_speech_features
tensorflow==1.4.0
PyYAML
tqdm
librosa
scikit-learn
```

### Install

We use miniconda for installing the dependencies, https://conda.io/miniconda.html
Create conda environment and install the dependencies in requirements.txt.

```
conda create --name speech python=3.6.2
source activate speech
pip install -U -r requirements.txt
```
If you come across any errors with conda install, you may use ```pip install -u *``` instead.  


### Run Training Phase

To train speech recognition we run python main with predefined parameters.

##### Numbers
```
python3 __main__.py --train --digit --dataset=./audio_numbers --config=src/config/lstm_ctc.yml
```
##### VCTK
```
python3 __main__.py --train --vctk --dataset={path_to_vctk_dataset}/VCTK-Corpus --config=src/config/lstm_ctc_VCTK.yml
```
##### Parameters
```
--train - proceds with training sequence
--digit - uses the digit training
--vctk - uses the vctk training
--dataset - path to training dataset
--config - given the config file for training
```

### Decoding
To use speech recognition we run python main with predefined parameters shown below.
Essential is to provided trained model saved in dictionary ```trained_model```, the path is provided in config file.
Training network with CTC function is time-consuming and in order to get the newest trained model, try to run ```git pull```.   
##### Numbers
```
python3 __main__.py --decode --digit --config=src/config/lstm_ctc.yml --decodefile={path_to_decode_wav_file}
```
##### VCTK
```
python3 __main__.py --decode --vctk --config=src/config/lstm_ctc_VCTK.yml --decodefile={path_to_decode_wav_file}
```
```
--decode - proceds with training sequence
--digit - uses the digit decoding
--vctk - uses the digit decoding
--config - path to config file of trained model
--decodefile - path to file which will be decoded
```
Concrete example of decoding
```
python __main__.py --decode --vctk --config=./src/config/lstm_ctc_VCTK.yml --decodefile=/Users/adamzvada/Desktop/VCTK-Corpus/wav48/p227/p227_001.wav
```

#### Tensorboard
For better monitoring and visualization of training phase run tensorboard command.
```
tensorboard --logdir ./tensorboard    
```
