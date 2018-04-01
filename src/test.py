import numpy as np
import matplotlib.pyplot as plt
from librosa import display
import python_speech_features as sf
from sklearn.preprocessing import scale

from src.DataSet import read_data_sets

num_classes = 10 # digits 0 - 9
path_to_training_data = '/Users/adamzvada/Documents/School/BP/SpeechRecognition/audio_numbers/'
filename = '/Users/adamzvada/Documents/School/BP/SpeechRecognition/audio_numbers'


dataset = read_data_sets(path_to_training_data)


# for audio in dataset.test.audios:
#     print(audio.sample_rate)


# audio_record = dataset.test.audios[0]
# mffc_feture = sf.mfcc(audio_record.data, audio_record.sample_rate)
# mffc_feture = np.swapaxes(mffc_feture,0,1)
# print(mffc_feture.shape)
#
# mfcc = scale(mffc_feture, axis=1)
#
#
#
# plt.figure(figsize=(12, 4))
# display.specshow(mfcc, sr=audio_record.sample_rate, x_axis='time', y_axis='mel')
# plt.title('MFCC')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

# delta_feat = sf.delta(mffc_feture, 13)
# plt.plot(delta_feat)
# plt.show()


# Datasets = collections.namedtuple("Datasets", ["train", "validation", "test"])
# Dataset = collections.namedtuple("Dataset", ["data", "target"])
#
#
# labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
# # load all train filenames
# train_list = []
# for label in labels:
#     files_list = os.listdir(os.path.join(path_to_training_data, label))
#     items = {}
#     items["label"] = int(label)
#     items["filenames"] = []
#     for filename in files_list:
#         items["filenames"].append(filename)
#     train_list.append(items)
#
#
# print(train_list[0]['filenames'])
#
#
# # load batch size of train_x and train_x
#
# batch_size = 8
#
# for i in range(batch_size):




# mfcc = speechpy.feature.mfcc(signal, sample_rate)
#
# plt.plot(mfcc)
# plt.show()

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape



def show_mfcc(sample_rate, signal):

    """"
        Given sample_rate and signal from audio. Calculates mfcc.
        Returns numpy array of feature vectors for window consitis of numcep(default 13) coef.
    """

    numcep = 13
    numcontext = 9

    mfcc_feat = sf.mfcc(signal, sample_rate)
    print(mfcc_feat.shape)
    mfcc_feat = mfcc_feat[::2]

    train_inputs = np.array([], np.float32)
    train_inputs.resize((mfcc_feat.shape[0], numcep + 2 * numcep * numcontext))



    mfcc_feat = np.swapaxes(mfcc_feat, 0, 1)

    mfcc_feat = scale(mfcc_feat, axis=1)


    plt.figure(figsize=(12/2, 4/2))
    display.specshow(mfcc_feat, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()




target_text = 'Zero one and two?'

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# deletes unwanted character and lower chars
original = ' '.join(target_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', '').replace("'", '').replace('!', '').replace('-', '')

# replace space(' ') on two spaces
targets = original.replace(' ', '  ')

# splits by words and spaces to array ['hello', '', 'how', '', 'are' ...]
targets = targets.split(' ')

# crates array of chars and instead of space('') puts <space> token
targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

# Transform char into index (A -> 1, B -> 2, ...)
targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                          for x in targets])

# Creating sparse representation to feed the placeholder
train_targets = sparse_tuple_from([targets])

print(train_targets)


# sample_rate, signal = wav.read(os.path.join(path_to_training_data, '0/0_jackson_0.wav'))
# show_mfcc(sample_rate, signal)
#
# sample_rate, signal = wav.read(os.path.join(path_to_training_data, '0/0_nicolas_0.wav'))
# show_mfcc(sample_rate, signal)
#
# sample_rate, signal = wav.read(os.path.join(path_to_training_data, '0/0_theo_0.wav'))
# show_mfcc(sample_rate, signal)
#
# plt.show()!')