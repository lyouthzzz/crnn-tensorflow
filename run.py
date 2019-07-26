import argparse
from crnn_net import CRNN

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Train or test the CRNN model.')

    parser.add_argument(
        "--train",
        action="store_true",
        help="Define if we train the model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we test the model"
    )
    parser.add_argument(
        "-ttr",
        "--train_test_ratio",
        type=float,
        nargs="?",
        help="How the data will be split between training and testing",
        default=0.70
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        default='./model_save/'
    )
    parser.add_argument(
        "-ex",
        "--examples_path",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
        required=True
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=64
    )
    parser.add_argument(
        "-epoch",
        "--epoch_count",
        type=int,
        nargs="?",
        help="How many epoch in training",
        default=10
    )
    parser.add_argument(
        "-miw",
        "--max_image_width",
        type=int,
        nargs="?",
        help="Maximum width of an example before truncating",
        default=100
    )

    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Define if we try to load a checkpoint file from the save folder"
    )

    return parser.parse_args()

def main():
    """
        Entry point when using CRNN from the commandline
    """

    args = parse_arguments()

    if not args.train and not args.test:
        print("If we are not training, and not testing, what is the point?")

    crnn = None

    if args.train:
        crnn = CRNN(
            args.batch_size,
            args.model_path,
            args.examples_path,
            args.max_image_width,
            args.train_test_ratio,
            args.restore
        )

        crnn.train(args.epoch_count)

    if args.test:
        if crnn is None:
            crnn = CRNN(
                args.batch_size,
                args.model_path,
                args.examples_path,
                args.max_image_width,
                0,
                args.restore
            )

        crnn.test()


if __name__ == '__main__':
    main()