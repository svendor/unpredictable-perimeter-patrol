from LSR.lsr_cycle import LSRCycle
from LSR.lsr_line import LSRLine

from jax_setup import optimize, print_human_readable

import numpy as np



if __name__ == "__main__":
    from cli import parse_args

    args = parse_args()
    n = args.n
    m = args.m
    num_iterations = args.num_iterations
    decay_rate = args.decay_rate
    topology = args.topology
    if topology == "line":
        strategy = LSRLine(n, m)
    elif topology == "cycle":
        strategy = LSRCycle(n, m)
    else:
        raise ValueError(f"Unknown topology: {topology}")


    my_template = strategy.get_transition_template()
    num_histories = strategy.get_num_histories()
    num_states = strategy.get_num_states()
    print(f"Number of histories: {num_histories}")
    print(f"Number of states: {num_states}")

    transition_matrix = optimize(my_template, num_states, num_histories, num_iterations, decay_rate)

    if args.show_strategy:
        strategy.print_human_readable(transition_matrix)

    if args.debug:
        strategy.print_transition_template(my_template)

