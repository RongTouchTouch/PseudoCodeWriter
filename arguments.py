import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--valid_file", default=None, type=str, required=True)
