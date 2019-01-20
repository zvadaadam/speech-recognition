import numpy as np
import tensorflow as tf
import re

SPACE_TOKEN = '<space>'

SPACE_INDEX = 0

FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


def get_refactored_transcript(txt, is_filename=True, is_digit=True):
    """"
        Fucntion reads text file and refactors its text. It extract unwanted characters(.,?,!,\n, etc.).
        Converts text to array for characters and map them to indexs.
        @:param txt_filename
        @:return array of characters mapped to integers from file text
    """


    if is_filename:
        txt = read_txt(txt)

    simple_text = ''
    if not is_digit:
        simple_text = simplify_text(txt)
    else:
        simple_text = simplify_text(text_number(txt))

    text_by_chars = text_to_chars(simple_text)

    return chars_to_index(text_by_chars)

def read_txt(filename):
    with open(filename) as f:
        txt = f.read()

    return txt

def chars_to_index(char_text):
    return np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in char_text])

def simplify_text(text):

    # lower text
    simplified_text = ' '.join(text.strip().lower().split(' '))

    # get ride of unwanted characters
    simplified_text = re.sub(r'([^\s[a-z])+', '', simplified_text)

    # converts multiple whitespaces to one space
    simplified_text = ' '.join(simplified_text.split())

    return simplified_text


def text_to_chars(text):
    # replace space(' ') on two spaces
    refactor_text = text.replace(' ', '  ')

    # splits by words and spaces to array ['hello', '', 'how', '', 'are' ...]
    refactor_text = refactor_text.split(' ')

    # crates array of chars and instead of space('') puts <space> token
    refactor_text = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in refactor_text])

    return refactor_text

def index_to_text(text_index):

    char_text = ''.join([chr(x) for x in np.asarray(text_index) + FIRST_INDEX])
    char_text = char_text.replace(chr(ord('z') + 1), '')
    char_text = char_text.replace(chr(ord('a') - 1), ' ')

    return char_text

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
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


def text_number(num):
    return {
         0 : 'zero',
         1 : 'one',
         2 : 'two',
         3 : 'three',
         4 : 'four',
         5 : 'five',
         6 : 'six',
         7 : 'seven',
         8 : 'eight',
         9 : 'nine'
    }[num]


def test():

    text_index = [1,0,19,13,1,12,12,0,14,21,13,2,5,18,0,15,6,0,14,5,23,0,10,15,2,19,0,23,9,12,12,0,1,12,19,15,0,2,5,0,3,18,5,1,20,5,4,-55]

    str = index_to_text(text_index)

    text = 'AHOJ,  jak se \n @ mas 1?'

    print(simplify_text(text))



if __name__ == "__main__":
    test()