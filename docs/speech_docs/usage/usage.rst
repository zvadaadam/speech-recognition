Usage
=====

Training
--------

Dataset
~~~~~~~

In order to train the model, you need to download some supported speech dataset.
Currently we supporte FreeSpokenDigit dataset and VCTK corpus.

Let's say we go with the VCTK dataset.
Download it from this `link <https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_.

The path to your locally stored dataset will be added to the configuration file::

    dataset:
        name: &dataset_name VCTK
        label_type: text
        lang: ENG
        dataset_path: {path/to/your/dataset/vctk}


Model Config
~~~~~~~~~~~~

Before running the training command, feel free set your own training configuration.

* Model
    - RNN
    - BRNN
* Number of layers
* Number of hidden cells
* Batch Size
* Number of Epoches
* Number of Iterations
* Dropout probability

Run
~~~
This is how you run the training process::

 $ python -m speechrecognition train --config {path/to/your/config-file}


Prediction
----------

The training process saves the trained model to directory defined in the configuration file.
You may use the model to run the predict on your own audio data::

 $ python -m speechrecognition predict --audio {path/to/audio-file} --config {path/to/your/config-file}

Not yet supported in the code!