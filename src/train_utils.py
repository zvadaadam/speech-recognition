from src import audio_utils, text_utils


def input_to_ctc_format(audio_filename, label_filename, numcep):

    audio_features = audio_utils.audio_to_feature_vectors(audio_filename, numcep)

    text_target = text_utils.get_refactored_transcript(label_filename, is_filename=False)

    sparse_target = text_utils.sparse_tuple_from(text_target)

    sequence_length = audio_utils.shape([1])

    return audio_features, text_target, sparse_target, sequence_length