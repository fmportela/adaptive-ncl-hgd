import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the experiment's configuration YAML file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the experiment directory if it exists.",
    )

    return parser
