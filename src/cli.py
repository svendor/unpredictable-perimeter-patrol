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
    parser.add_argument(
        "-t",
        "--topology",
        type=str,
        default="line",
        help="The network topology to use (default: line).",
    )
    parser.add_argument(
        "-s",
        "--show_strategy",
        action="store_true",
        help="Whether to print the strategy (default: False).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print debug information (default: False).",
    )
    parser.add_argument(
        "-i",
        "--num_iterations",
        type=int,
        default=400,
        help="The number of iterations to run (default: 400).",
    )
    parser.add_argument(
        "-d",
        "--decay_rate",
        type=float,
        default=0.5,
        help="The decay rate of the learning, decreases step size after a number of iterations to make smaller adjustments when approaching the optimal solution. Should be a value between 0 and 1 (default: 0.5).",
    )
    return parser.parse_args()