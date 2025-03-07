"""
CSC 242 - Intro to AI
Project 2 - Sudoku
Christopher DelGuercio (cdelguer@u.rochester.edu)
Krish Patel (kpatel46@u.rochester.edu)
"""

import sys
from collections import deque

## FULL_DOMAIN is 511 in decimal but 111111111 in binary
## each bit represents a number from 1-9 which we will utilize when assigning the domain of each cell
FULL_DOMAIN = (1 << 9) - 1


def bit_to_values(mask: int) -> list[int]:
    """
        This method converts the bitmask into a list of numbers that are represented by the bits
        into a list of numbers from 1-9

        params:
            mask: int - the bitmask that we want to convert into a list of numbers

        returns:
            list[int] - a list of numbers that are represented by the bits in the mask
    """
    return [i + 1 for i in range(9) if mask & (1 << i)]


class SudokuBoard:
    def __init__(self, input_grid: list[list[int]]):
        self.size = 9
        self.subgrid_size = 3
        self.grid = [row[:] for row in input_grid]

        # the domain is represented by a bitmask where each bit represents a number from 1-9
        # if cell is empty then we go with the full_domain (111111111)
        self.domains = [
            [FULL_DOMAIN if self.grid[row_index][col_index] == 0 else (1 << (self.grid[row_index][col_index] - 1))
            for col_index in range(self.size)]
            for row_index in range(self.size)
        ]

        # Precompute all neighbors for each cell in the grid
        self.neighbors = {}
        for row_index in range(self.size):
            for col_index in range(self.size):
                neighbor_cells = set()

                # Rows
                for other_col in range(self.size):
                    if other_col != col_index:
                        neighbor_cells.add((row_index, other_col))

                # Columns
                for other_row in range(self.size):
                    if other_row != row_index:
                        neighbor_cells.add((other_row, col_index))

                # Subgrid
                subgrid_row = (row_index // self.subgrid_size) * self.subgrid_size
                subgrid_col = (col_index // self.subgrid_size) * self.subgrid_size
                for sub_row in range(subgrid_row, subgrid_row + self.subgrid_size):
                    for other_col in range(subgrid_col, subgrid_col + self.subgrid_size):
                        if (sub_row, other_col) != (row_index, col_index):
                            neighbor_cells.add((sub_row, other_col))

                self.neighbors[(row_index, col_index)] = tuple(neighbor_cells)

    def is_complete(self) -> bool:
        """
            This method just checks if the entire grid is filled in -- no zeros remaining

            params:
                None

            returns:
                bool - True if the grid is completely filled in, False otherwise
        """
        return all(self.grid[row_index][col_index] != 0 for row_index in range(self.size) for col_index in range(self.size))

    def assign_value(self, row: int, col: int, value: int):
        """
            This method assigns a val to the specified cell in the grid and updates the relavent domains

            params:
                row: int - the row of the cell we want to assign a value to
                col: int - the column of the cell we want to assign a value to
                value: int - the value we want to assign to the cell

            returns:
                None
        """
        self.grid[row][col] = value
        self.domains[row][col] = 1 << (value - 1)

    def remove_value(self, row: int, col: int):
        """
            This method removes a value from the specified cell in the grid and updates the relavent domains

            params:
                row: int - the row of the cell we want to remove a value from
                col: int - the column of the cell we want to remove a value from

            returns:
                None
        """
        self.grid[row][col] = 0
        self.domains[row][col] = FULL_DOMAIN

    def display_board(self):
        """
            this method simply prints out the grid
        """
        for board_row in self.grid:
            print(" ".join(str(num) for num in board_row))


def is_valid_assignment(board: SudokuBoard, row: int, col: int, value: int) -> bool:
    """
        This method checks if assigning a value to a cell is valid
        by only using the precomputed neighbors

        params:
            board: SudokuBoard - the board we are working with
            row: int - the row of the cell we want to assign a value to
            col: int - the column of the cell we want to assign a value to
            value: int - the value we want to assign to the cell

        returns:
            bool - True if the assignment is valid, False otherwise
    """
    for (neighbor_row, neighbor_col) in board.neighbors[(row, col)]:
        if board.grid[neighbor_row][neighbor_col] == value:
            return False
    return True


# AC3 revise: if cell2 has a value already, remove that bit from cell1’s domain.
def revise(board: SudokuBoard, cell1: tuple[int, int], cell2: tuple[int, int]) -> bool:
    """
        this method checks if the domain of 'cell2' is just one number (single bit)
        if it is, then we remove that number from the domain of 'cell1'

        params:
            board: SudokuBoard - the board we are working with
            cell1: tuple[int, int] - the cell we want to revise
            cell2: tuple[int, int] - the cell we are comparing to

        returns:
            bool - True if we revised the domain of cell1
    """
    row1, col1 = cell1
    row2, col2 = cell2
    revised = False
    domain2 = board.domains[row2][col2]

    # we can check if dom2 is a singleton by checking if the bitwise AND of dom2 and dom2 - 1 is 0
    # this should work because if dom2 is a power of 2, then it will only have one bit set to 1
    if domain2 and (domain2 & (domain2 - 1)) == 0:
        if board.domains[row1][col1] & domain2:
            board.domains[row1][col1] &= ~domain2
            revised = True
    return revised


def ac3(board: SudokuBoard) -> bool:
    """
        this method implements the AC3 algorithm to solve the sudoku puzzle
        it uses a queue to keep track of all the arcs that need to be checked
        and then uses the revise method to check if the domains need to be revised

        params:
            board: SudokuBoard - the board we are working with

        returns:
            bool - True if the puzzle is solvable, false otherwise
    """
    arc_queue = deque()
    for row in range(board.size):
        for col in range(board.size):
            for neighbor_cell in board.neighbors[(row, col)]:
                arc_queue.append(((row, col), neighbor_cell))
    while arc_queue:
        current_arc = arc_queue.popleft()
        current_cell, compared_cell = current_arc
        if revise(board, current_cell, compared_cell):
            current_row, current_col = current_cell
            if board.domains[current_row][current_col] == 0:
                return False
            for neighbor_cell in board.neighbors[(current_row, current_col)]:
                if neighbor_cell != compared_cell:
                    arc_queue.append((current_cell, neighbor_cell))
    return True


def incremental_propagate(board: SudokuBoard, start_cell: tuple[int, int]) -> (bool, dict):
    """
        this method implements the incremental propagation algorithm to solve the sudoku puzzle
        it uses a queue to keep track of all the cells that need to be checked
        and then uses the revise method to check if the domains need to be revised similar to the AC3 method
        This method however also keeps track of the domains and grid values that were changed
    """
    # saved: maps cell -> (old_domain, old_grid)
    saved = {}
    prop_queue = deque([start_cell])
    while prop_queue:
        current_row, current_col = prop_queue.popleft()
        assigned_mask = board.domains[current_row][current_col]
        # Propagate the fact that (current_row,current_col) is assigned.
        for neighbor in board.neighbors[(current_row, current_col)]:
            neighbor_row, neighbor_col = neighbor
            if board.domains[neighbor_row][neighbor_col] & assigned_mask:
                if neighbor not in saved:
                    saved[neighbor] = (board.domains[neighbor_row][neighbor_col], board.grid[neighbor_row][neighbor_col])
                board.domains[neighbor_row][neighbor_col] &= ~assigned_mask
                if board.domains[neighbor_row][neighbor_col] == 0:
                    return False, saved
                # If this neighbor becomes singleton and is not yet assigned, assign it.
                if board.domains[neighbor_row][neighbor_col].bit_count() == 1 and board.grid[neighbor_row][neighbor_col] == 0:
                    val = board.domains[neighbor_row][neighbor_col].bit_length()  # For power-of-two, bit_length() gives value.
                    board.grid[neighbor_row][neighbor_col] = val
                    prop_queue.append((neighbor_row, neighbor_col))
    return True, saved


def restore_domains(board: SudokuBoard, saved: dict):
    """
        this method assists in restoring the last saved domains and grid values
        in essence, revert to the last state that was possible
    """
    for (row, col), (old_domain, old_grid) in saved.items():
        board.domains[row][col] = old_domain
        board.grid[row][col] = old_grid


def get_unassigned_variable(board: SudokuBoard):
    """
        this method gets the next unassigned variable in the grid using the minimum remaining val
        heuristic--which cell has the fewest remaining values in its domain
    """
    min_count = 10
    chosen = None
    for row in range(board.size):
        for col in range(board.size):
            if board.grid[row][col] == 0:
                count = board.domains[row][col].bit_count()
                if count < min_count:
                    min_count = count
                    chosen = (row, col)
    return chosen


def get_sorted_domain_values(board: SudokuBoard, row: int, col: int) -> list[int]:
    """
        this method is helps with evaluating the least constraining value heuristic
        by sorting the values in the domain of the cell based on the number of conflicts
        that each value would introduce
    """
    mask = board.domains[row][col]
    values = bit_to_values(mask)
    return sorted(values)


# Backtracking search with MRV, LCV, and incremental propagation.
def backtrack(board: SudokuBoard) -> bool:
    """
        this method links together the usage of the MRV, LCV and incremental propogation methods.
        It uses a recursive backtracking algorithm by assigning values
        to the cells and checking if the assignment is valid. If it is, it will continue assigning
        values to the next cell and so on. If it reaches a point where it cannot assign a value to a cell
        it will backtrack and try a different value
    """
    if board.is_complete():
        return True
    chosen_cell = get_unassigned_variable(board)
    if chosen_cell is None:
        return False
    current_row, current_col = chosen_cell
    possible_values = get_sorted_domain_values(board, current_row, current_col)
    original_domain = board.domains[current_row][current_col]
    original_grid_value = board.grid[current_row][current_col]
    for candidateValue in possible_values:
        if is_valid_assignment(board, current_row, current_col, candidateValue):
            board.assign_value(current_row, current_col, candidateValue)
            prop_success, changes_saved = incremental_propagate(board, (current_row, current_col))
            if prop_success and backtrack(board):
                return True
            restore_domains(board, changes_saved)
            board.domains[current_row][current_col] = original_domain
            board.grid[current_row][current_col] = original_grid_value
    return False


def read_input() -> list[list[int]]:
    """
        input stuff
    """
    grid = []
    for _ in range(9):
        line = sys.stdin.readline().strip()
        if not line:
            break
        grid.append(list(map(int, line.split())))
    return grid


def write_output(board: SudokuBoard) -> None:
    """
        this method writes the output of the sudoku puzzle to the console or prints No solution
        if the puzzle is not solvable.

        params:
            board: SudokuBoard - the board we are working with

        returns:
            None
    """
    if board.is_complete():
        board.display_board()
    else:
        print("No solution.")


def main():
    grid = read_input()
    board = SudokuBoard(grid)
    solution_found = ac3(board) and backtrack(board)
    if solution_found:
        write_output(board)


if __name__ == '__main__':
    main()