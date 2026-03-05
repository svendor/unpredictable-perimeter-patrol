from abc import ABC, abstractmethod

class Strategy(ABC):

    @abstractmethod
    def get_transition_template(self) -> list[list[tuple[int, int]]]:
        """
        Return a list of a list of tuples (i,j) such that state i can transition to state j.

        Each inner list can be seen as a list of equivalent moves.
        That is, the probability of transitioning from state i to state j is the same for all (i,j) in the same inner list.
        """
        pass

    @abstractmethod
    def get_history_string(self, history_index: int) -> str:
        """
        Return a human-readable string representation of the history corresponding to the given index.
        This is useful for debugging and understanding the strategy.
        """
        pass