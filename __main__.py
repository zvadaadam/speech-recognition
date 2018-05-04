import sys
from src.train import train_new


if __name__ == "__main__":

    print(sys.path)

    args = sys.argv
    if len(args) == 2:
        train_new.main(config_path=args[1])
    elif len(args) == 3:
        train_new.main(config_path=args[1], dataset_path=args[2])
    else:
        train_new.main()

