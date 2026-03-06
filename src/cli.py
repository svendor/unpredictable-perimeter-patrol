import argparse

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find the optimal strategy for the patroller on a cycle."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=4,
        help="The number of locations on the cycle (default: 4).",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=1,
        help="The memory length of the patroller's history (default: 1).",
    )
    return parser.parse_args()