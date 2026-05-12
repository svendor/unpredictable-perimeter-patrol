# from strategy import Strategy
from LSR.lsr_history import LSRHistory
from strategy import Strategy
import numpy as np

class LSRCycle(Strategy):
    """
    Contains a list of triples (cur, left, right) for each history,
    where the entries are probabilities of moving left, right or staying.
    The index of the probability corresponds to the move: 0 for same, 1 for left, 2 for right.
    """

    def __init__(self, n, m):
        """
        n is the number of locations
        m is the number of locations that is remembered, thus the history has length m-1 in this case.
        """
        self.n = n
        self.m = m
        self.num_histories = LSRHistory.num_histories(m)


    def get_num_states(self) -> int:
        return self.n * self.num_histories

    def get_num_histories(self) -> int:
        return self.num_histories

    def get_num_vertices(self) -> int:
        return self.n


    def get_history_string(self, history_index: int) -> str:
        return f"{LSRHistory.from_index(history_index, self.m)}"


    def get_transition_template(self) -> list[list[tuple[int, int]]]:
        output = []

        for triples in LSRHistory.get_transitions_per_history(self.m):
            cur_list = []
            for history_index, next_history_index, move in triples:
                vertices = np.arange(self.n)
                if move == LSRHistory.SAME:
                    next_vertices = vertices
                elif move == LSRHistory.LEFT:
                    next_vertices = (vertices - 1) % self.n
                else:
                    next_vertices = (vertices + 1) % self.n

                start_index = vertices * self.num_histories + history_index
                end_index = next_vertices * self.num_histories + next_history_index
                cur_list += list(zip(start_index, end_index))

            output.append(cur_list)

        return output

    def print_human_readable(self, transition_matrix):
        """Take the optimized parameters and print them in a human-readable format."""
        rows, cols, params = transition_matrix.indices[:, 0], transition_matrix.indices[:, 1], transition_matrix.data

        print("\nOptimized Transition Probabilities from location 0:")
        num_histories = self.get_num_histories()
        n = self.get_num_vertices()

        # We know that, by symmetry, we only need to consider location 0 and all their histories.
        # In other words, only the rows indexed by 0, 1, ..., num_histories-1 are relevant.
        moves = np.zeros((num_histories, 3), dtype=float)

        for i, j, parameter in zip(rows, cols, params):
            if i < num_histories:
                next_loc = j // num_histories
                # next_loc is n-1, 0 or 1, if the moves were Left Same or Right. By adding 1 and performing modulo
                # we obtain 0, 1, 2.
                move_index = (next_loc + 1) % n
                moves[i, move_index] = parameter

        for i in range(num_histories):
            print(f"{self.get_history_string(i)}: {moves[i]}")

    def print_transition_template(self, template: list[list[tuple[int, int]]]):
        for cur_list in template:
            print("New list:")
            for start, end in cur_list:
                sloc, shist = divmod(start, self.num_histories)
                eloc, ehist = divmod(end, self.num_histories)
                print(f"  {sloc}: {self.get_history_string(shist)} -> {eloc}: {self.get_history_string(ehist)}")


if __name__ == "__main__":
    cycle = LSRCycle(3, 2)
    output = cycle.get_transition_template()
    for cur_list in output:
        print("New list:")
        for start, end in cur_list:
            print(f"  {start} -> {end}")