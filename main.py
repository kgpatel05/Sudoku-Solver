import numpy as np

class SudokuBoard():

    def __init__(self, input_grid: list[list[int]]):
        '''
            Initializes our Sudoku Board

            params:
                input_grid: list[list[int]] - A 2D list representing the initial state of the board.
        '''
        self.size = 9
        self.subgrid_size: int = 3
        self.grid = np.array(input_grid)
        self.constraints = self._generate_constraints()

        # init domains as a 2D list of sets
        self.domains = [[set() for _ in range(self.size)] for _ in range(self.size)]
        for row in range(self.size):
            for col in range(self.size):
                if input_grid[row][col] == 0:
                    self.domains[row][col] = set(range(1, 10))
                else:
                    self.domains[row][col] = {input_grid[row][col]}

    def _generate_constraints(self) -> set[tuple[int. int]]:
        # TODO: horrible time complexity...any way to optimize this?
        '''
            Internal method to help construct tthe constraints for the board.
            Our constraint provides a set of tuples, where each tuple represents a pair of cells that
            cannot share the same value

            params:
                None

            returns:
                set[tuple[int, int]] - A set of tuples representing the constraints of the board.
        '''

        constraints = set()

        for row in range(self.size):
            for col in range(self.size):

                #row constraints
                for r in range(self.size):
                    if r != row:
                        constraints.add(((row, col), (r, col)))

                #column constraints
                for c in range(self.size):
                    if c != col:
                        constraints.add(((row, col), (row, c)))

                #subgrid constraints
                start_row = (row // self.subgrid_size) * self.subgrid_size
                start_col = (col // self.subgrid_size) * self.subgrid_size
                for r in range(start_row, start_row + self.subgrid_size):
                    for c in range(start_col, start_col + self.subgrid_size):
                        if (r, c) != (row, col):
                            constraints.add(((row, col), (r, c)))

        return constraints

                




    def is_complete(self) -> bool:
        """
            This method DOES NOT return if the board is currently solved rather
            it simply indicates if every position is occupied and hence the board
            is complete.

            params:
                None

            returns:
                bool - True if the board is complete, False otherwise.
        """
        return not np.any(self.grid == 0)

    def is_valid_assignment(self, row: int, col: int, value: int) -> bool:
        pass

    def assign_value(self, row: int, col: int, value: int) -> None:
        pass

    def remove_value(self, row: int, col: int) -> None:
        pass

    def display_board(self) -> None:
        pass


def ac3(board: SudokuBoard) -> bool:
    pass 


def get_unassigned_variable(board: SudokuBoard) -> tuple[int, int]:
    pass

def get_sorted_domain_values(board: SudokuBoard, row: int, col: int) -> list[int]:
    pass

def backtrack(board: SudokuBoard) -> bool:
    pass

def read_input() -> list[list[int]]:
    pass

def write_output(board: SudokuBoard) -> None:
    pass

def main():
    """
        General Pipeline:
        Read input. (read_input())
        Initialize board. (SudokuBoard)
        Apply AC-3. (ac3(board))
        Use backtracking search. (backtrack(board))
        Output solution. (write_output(board.grid))
    """
    pass