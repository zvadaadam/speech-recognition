import sys
from src.train import train_new


if __name__ == "__main__":

    args = sys.argv
    if len(args) == 2:
        train_new.main(config_path=args[1])
    else:
        train_new.main()

