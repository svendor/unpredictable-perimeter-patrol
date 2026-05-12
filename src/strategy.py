from abc import ABC, abstractmethod

class Strategy(ABC):

    def print_transition_template(self, template: list[list[tuple[int, int]]]):
        for cur_list in template:
            print("New list:")
            for start, end in cur_list:
                print(f"  {start} -> {end}")


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


    @abstractmethod
    def get_num_states(self) -> int:
        """
        Return the total number of states in the strategy,
        which is typically the number of locations multiplied by the number of histories.
        """
        pass


    @abstractmethod
    def get_num_histories(self) -> int:
        """
        Return the number of histories in the strategy.
        """
        pass


    def get_num_vertices(self) -> int:
        """
        Return the number of vertices in the strategy
        """
        pass


    def print_human_readable(self, transition_matrix) -> None:
        """
        Prints a human readable description of the strategy
        """
        pass