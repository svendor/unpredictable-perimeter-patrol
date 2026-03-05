# from strategy import Strategy
from LSR.lsr_history import LSRHistory
import numpy as np

class LSRCycle():
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


    def get_history_string(self, history_index: int) -> str:
        return f"{LSRHistory.from_index(history_index, self.m)}"


    def get_transition_template(self) -> list[list[tuple[int, int]]]:
        output = []

        for triples in self.__get_transitions_per_history():
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


    def __get_transitions_per_history(self) -> list[list[tuple[int, int, int]]]:
        """
        Returns a list of lists.

        An inner list contains equivalent transitions. That is, transitions that should occur with the same probability.
        A transition is characterized by three integers: the current history index, the next history index and the move that causes the transition.
        """

        output = []

        seen_indices = np.zeros(self.num_histories, dtype=bool)

        for history_index in range(self.num_histories):
            # We only need to consider one of the two flipped histories, since they have the same probabilities.
            flipped_index = LSRHistory.flipped_index(history_index, self.m)

            if seen_indices[flipped_index]:
                continue

            seen_indices[history_index] = True
            seen_indices[flipped_index] = True

            # Consider move SAME
            cur_list = []
            next_history_index = LSRHistory.next_index(history_index, LSRHistory.SAME, self.m)
            cur_list.append((history_index, next_history_index, LSRHistory.SAME))
            if flipped_index != history_index:
                cur_list.append((flipped_index, LSRHistory.next_index(flipped_index, LSRHistory.SAME, self.m), LSRHistory.SAME))
            output.append(cur_list)
            # print(output[-1])

            # Consider moves LEFT and RIGHT
            cur_list = []
            next_history_index = LSRHistory.next_index(history_index, LSRHistory.LEFT, self.m)
            cur_list.append((history_index, next_history_index, LSRHistory.LEFT))
            cur_list.append((flipped_index, LSRHistory.next_index(flipped_index, LSRHistory.RIGHT, self.m), LSRHistory.RIGHT))
            output.append(cur_list)
            # print(output[-1])

            if flipped_index != history_index:
                cur_list = []
                next_history_index = LSRHistory.next_index(history_index, LSRHistory.RIGHT, self.m)
                cur_list.append((history_index, next_history_index, LSRHistory.RIGHT))
                cur_list.append((flipped_index, LSRHistory.next_index(flipped_index, LSRHistory.LEFT, self.m), LSRHistory.LEFT))
                output.append(cur_list)
                # print(output[-1])



        return output



if __name__ == "__main__":
    cycle = LSRCycle(3, 2)
    output = cycle.get_transition_template()
    for cur_list in output:
        print("New list:")
        for start, end in cur_list:
            print(f"  {start} -> {end}")