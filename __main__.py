import sys
from src.train import train
from src import decoder
from optparse import OptionParser
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()


    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--decode', action="store_true", default=False)
    parser.add_argument('--digit', action="store_true", default=False)
    parser.add_argument('--vctk', action="store_true", default=False)
    parser.add_argument('--config', action="store", default='./src/config/lstm_ctc_VCTK.yml')
    parser.add_argument('--model', action="store", default='./trained_models/three_speaker_model-147')
    parser.add_argument('--decodefile', action="store", default='/Users/adamzvada/Desktop/VCTK-Corpus/wav48/p225/p225_001.wav')
    parser.add_argument('--dataset', action="store", default='/Users/adamzvada/Desktop/VCTK-Corpus')


    args = parser.parse_args()

    is_training = args.train
    is_decoding = args.decode
    is_digit = args.digit
    is_vctk = args.vctk
    config_file = args.config
    model_file = args.model
    input_file = args.decodefile
    dataset_file = args.dataset


    if is_digit == is_vctk:
        sys.exit("Cannot perfrom digits and vctk simultaneously...")

    if is_training == is_decoding:
        sys.exit("Cannot train and decode simultaneously...")


    if is_training:
       train.main(config_file, dataset_file, is_vctk)

        # args = sys.argv
        # if len(args) == 2:
        #     train_new.main(config_path=args[1])
        # elif len(args) == 3:
        #     train_new.main(config_path=args[1], dataset_path=args[2])
        # else:
        #     train_new.main()
    else:
       decoder.predict(input_file, config_file)
