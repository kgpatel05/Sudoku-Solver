import numpy as np

class SudokuBoard():

    def __init__(self, input_grid: list[list[int]]):
        grid: list[list[int]] = None
        domains: list[list[set[int]]] = None
        constraints: set[tuple[int, int]] = None
        size: int = 9
        subgrid_size: int = 3

    def is_complete(self) -> bool:
        pass

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