import numpy as np

class LSRHistory():
    SAME = 0
    LEFT = 1
    RIGHT = 2

    """
    The history is a sequence of length m-1 of the previous moves of the patroller. This models the last 3 locations in a more data-efficient manner.
    The history is encoded as a base-3 number, where the least significant digit is the most recent move. In the array this is the last index.
    """

    def __init__(self, m: int, history: np.array =None):
        self.m = m
        if history is None or len(history) != m-1:
            self.history = np.zeros(m-1, dtype=int)
        else:
            self.history = history

    def flip_move(move: int) -> int:
        if move == LSRHistory.LEFT:
            return LSRHistory.RIGHT
        elif move == LSRHistory.RIGHT:
            return LSRHistory.LEFT
        else:
            return LSRHistory.SAME

    def flipped(self):
        """
        Returns a new history object, where LEFT and RIGHT are replaced.
        """
        new_history = np.array(
            [LSRHistory.flip_move(move) for move in self.history]
        )
        return LSRHistory(self.m, new_history)

    def flipped_index(index: int, m: int) -> int:
        history = LSRHistory.from_index(index, m)
        flipped_history = history.flipped()
        return flipped_history.to_index()

    def to_index(self) -> int:
        if self.m == 1:
            return 0
        return np.dot(self.history, 3**np.arange(len(self.history)))

    def from_index(index: int, m: int):
        assert index < 3**(m-1)
        return LSRHistory(m, np.array([int((index // 3**i) % 3) for i in range(m-1)]))

    def next_history(self, move: int):
        new_history = self.history[1:]
        new_history = np.append(new_history, move)
        return LSRHistory(self.m, new_history)

    def next_index(index: int, move: int, m: int) -> int:
        if m == 1:
            return 0
        return (index // 3) + move * 3**(m-2)

    def num_histories(m: int) -> int:
        return 3**(m-1)


    def __to_letter(self, move: int):
        return "L" if move == LSRHistory.LEFT else "R" if move == LSRHistory.RIGHT else "S"


    def __repr__(self):
        return "".join([self.__to_letter(move) for move in self.history])


    def get_transitions_per_history(m: int) -> list[list[tuple[int, int, int]]]:
        """
        Returns a list of lists.

        An inner list contains equivalent transitions. That is, transitions that should occur with the same probability.
        A transition is characterized by three integers: the current history index, the next history index and the move that causes the transition.
        """

        output = []
        num_histories = LSRHistory.num_histories(m)
        seen_indices = np.zeros(num_histories, dtype=bool)

        for history_index in range(num_histories):
            # We only need to consider one of the two flipped histories, since they have the same probabilities.
            flipped_index = LSRHistory.flipped_index(history_index, m)

            if seen_indices[flipped_index]:
                continue

            seen_indices[history_index] = True
            seen_indices[flipped_index] = True

            # Consider move SAME
            cur_list = []
            next_history_index = LSRHistory.next_index(history_index, LSRHistory.SAME, m)
            cur_list.append((history_index, next_history_index, LSRHistory.SAME))
            if flipped_index != history_index:
                cur_list.append((flipped_index, LSRHistory.next_index(flipped_index, LSRHistory.SAME, m), LSRHistory.SAME))
            output.append(cur_list)
            # print(output[-1])

            # Consider moves LEFT and RIGHT
            cur_list = []
            next_history_index = LSRHistory.next_index(history_index, LSRHistory.LEFT, m)
            cur_list.append((history_index, next_history_index, LSRHistory.LEFT))
            cur_list.append((flipped_index, LSRHistory.next_index(flipped_index, LSRHistory.RIGHT, m), LSRHistory.RIGHT))
            output.append(cur_list)
            # print(output[-1])

            if flipped_index != history_index:
                cur_list = []
                next_history_index = LSRHistory.next_index(history_index, LSRHistory.RIGHT, m)
                cur_list.append((history_index, next_history_index, LSRHistory.RIGHT))
                cur_list.append((flipped_index, LSRHistory.next_index(flipped_index, LSRHistory.LEFT, m), LSRHistory.LEFT))
                output.append(cur_list)
                # print(output[-1])


        return output
