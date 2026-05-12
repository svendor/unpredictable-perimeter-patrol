# from strategy import Strategy
from LSR.lsr_history import LSRHistory
from strategy import Strategy
import numpy as np



class LSRLine(Strategy):
    MIDDLE_SYMMETRY=0 # The symmetry is as follows abcddcba, where identical letters imply symmetry. This symmetry is always forced
    ALL_INNER_EQUAL=1 # The symmetry is as follows abbbbbba, where identical letters imply symmetry

    """
    Contains a list of triples (cur, left, right) for each history,
    where the entries are probabilities of moving left, right or staying.
    The index of the probability corresponds to the move: 0 for same, 1 for left, 2 for right.
    """

    def __init__(self, n, m, symmetry_type=ALL_INNER_EQUAL):
        """
        n is the number of locations
        m is the number of locations that is remembered, thus the history has length m-1 in this case.
        """
        self.n = n
        self.m = m
        self.num_histories = LSRHistory.num_histories(m)
        self.symmetry_type = symmetry_type


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
            output += self.__get_symmetries_from_triples(triples)

        return list(list(set(x)) for x in output)


    def __get_middle_symmetry_classes(self) -> list[np.array]:
        return [np.arange((self.n+1)//2)]


    def __get_all_inner_equal_classes(self) -> list[np.array]:
        return [np.arange(1), np.arange(1, (self.n+1)//2)]


    def __get_symmetry_classes(self) -> list[np.array]:
        """
        Returns a list of lists. Each inner lists is a class of vertices that are symmetric. Since the MIDDLE_SYMMETRY is forced, we only give the first n+1//2 vertices.
        """
        if self.symmetry_type == LSRLine.MIDDLE_SYMMETRY:
            return self.__get_middle_symmetry_classes()
        elif self.symmetry_type == LSRLine.ALL_INNER_EQUAL:
            return self.__get_all_inner_equal_classes()
        else:
            raise ValueError(f"Unknown symmetry type: {self.symmetry_type}")


    def __get_vertex_adjacencies(self, vertices: np.array, move: int) -> tuple[np.array, np.array]:
        # Compute where the next vertex is, given that the last move was move.
        next_vertices = vertices.copy()
        if move == LSRHistory.LEFT:
            next_vertices -= 1
        elif move == LSRHistory.RIGHT:
            next_vertices += 1

        # Remove transitions that are impossible (0 -> n-1 when LEFT, n-1 -> 0 when RIGHT).
        mask = next_vertices >= 0
        mask &= next_vertices < self.n
        vertices = vertices[mask]
        next_vertices = next_vertices[mask]

        return vertices, next_vertices


    def __get_symmetries_from_triples(self, triples: list[tuple[int, int, int]]) -> list[list[tuple[int, int]]]:
        """
        On input of a set of equivalent triples, output equivalent transitions, where the vertices are now included in the state.
        """
        classes = self.__get_symmetry_classes()
        output = [[] for _ in range(len(classes))]

        for history_index, next_history_index, move in triples:
            flipped_index = LSRHistory.flipped_index(history_index, self.m)
            flipped_next_index = LSRHistory.flipped_index(next_history_index, self.m)

            for i, vertices in enumerate(classes):
                vertices, next_vertices = self.__get_vertex_adjacencies(vertices, move)

                start_index = vertices * self.num_histories + history_index
                end_index = next_vertices * self.num_histories + next_history_index

                output[i] += list(zip(start_index, end_index))

                flipped_vertices = (self.n-1) - vertices
                flipped_next_vertices = (self.n-1) - next_vertices

                start_index = flipped_vertices * self.num_histories + flipped_index
                end_index = flipped_next_vertices * self.num_histories + flipped_next_index

                output[i] += list(zip(start_index, end_index))


        return output

    def print_human_readable(self, transition_matrix):
        """Take the optimized parameters and print them in a human-readable format."""
        rows, cols, params = transition_matrix.indices[:, 0], transition_matrix.indices[:, 1], transition_matrix.data

        classes = self.__get_symmetry_classes()

        for c in classes:
            representative = c[0]

            print("\nOptimized Transition Probabilities for class:", c)
            num_histories = self.get_num_histories()
            n = self.get_num_vertices()

            # We know that, by symmetry, we only need to consider the representative location and all their histories.
            # In other words, only the rows indexed by r//num_histories == c[0] are relevant.
            moves = np.zeros((num_histories, 3), dtype=float)

            def get_move(loc, next_loc):
                if loc == next_loc:
                    return 1
                elif loc == (next_loc + 1) % n:
                    return 0
                else:
                    return 2

            for i, j, parameter in zip(rows, cols, params):
                if i//num_histories == c[0]:
                    next_loc = j // num_histories
                    i %= num_histories
                    move_index = get_move(c[0], next_loc)
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
    line = LSRLine(3, 2)
    output = line.get_transition_template()

    for cur_list in output:
        print("New list:")
        for start, end in cur_list:
            sloc, shist = divmod(start, line.num_histories)
            eloc, ehist = divmod(end, line.num_histories)
            print(f"  {sloc}: {line.get_history_string(shist)} -> {eloc}: {line.get_history_string(ehist)}")

    # cycle = LSRCycle(3, 2)

    # output = cycle.get_transition_template()
    # for cur_list in output:
    #     print("New list:")
    #     for start, end in cur_list:
    #         print(f"  {start} -> {end}")