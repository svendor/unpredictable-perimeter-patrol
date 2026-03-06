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
        return (index // 3) + move * 3**(m-2)

    def num_histories(m: int) -> int:
        return 3**(m-1)


    def __to_letter(move: int):
        return "L" if move == LSRHistory.LEFT else "R" if move == LSRHistory.RIGHT else "S"


    def __repr__(self):
        return "".join([LSRHistory.__to_letter(move) for move in self.history])