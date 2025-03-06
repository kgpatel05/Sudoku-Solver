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

        #the domain is represented by a bitmask where each bit represents a number from 1-9
        # if cell is empty then we go with teh full_domain (111111111)
        self.domains = [
            [FULL_DOMAIN if self.grid[r][c] == 0 else (1 << (self.grid[r][c] - 1))
            for c in range(self.size)]
            for r in range(self.size)
        ]

        # Precompute neighbors for each cell (all cells in the same row, column, and 3x3 subgrid).
        #We precompute all of the neighbors for each cell in the grid
        self.neighbors = {}
        for row in range(self.size):
            for col in range(self.size):
                neighbors = set()
                # Rows
                for column in range(self.size):
                    if column != col:
                        neighbors.add((row, column))
                # Columns
                for row_ in range(self.size):
                    if row_ != row:
                        neighbors.add((row_, col))
                # Subgrid
                br = (row // self.subgrid_size) * self.subgrid_size
                bc = (col // self.subgrid_size) * self.subgrid_size
                for row_ in range(br, br + self.subgrid_size):
                    for column in range(bc, bc + self.subgrid_size):
                        if (row_, column) != (row, col):
                            neighbors.add((row_, column))
                self.neighbors[(row, col)] = tuple(neighbors)

    def is_complete(self) -> bool:
        """
            This method jsut checks if the entire grid is filled in -- no zeros remaining

            params:
                None
            
            returns:
                bool - True if the grid is completely filled in, False otherwise
        """
        return all(self.grid[row][col] != 0 for row in range(self.size) for col in range(self.size))

    def assign_value(self, row: int, col: int, value: int):
        """
            This mehtod assigns a val to the specified cell in the grid and updates the relavent domains

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
            This method removes a valuye from the specified cell in the grid and updates the relavent domains

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
        for row in self.grid:
            print(" ".join(str(x) for x in row))

def is_valid_assignment(board: SudokuBoard, row: int, col: int, value: int) -> bool:
    """
        This method checks if assigning a value to a cell is valid
        by only usingthe precomputed neighbors

        params:
            board: SudokuBoard - the board we are working with
            row: int - the row of the cell we want to assign a value to
            col: int - the column of the cell we want to assign a value to
            value: int - the value we want to assign to the cell

        returns:
            bool - True if the assignment is valid, False otherwise
    """
    for (r, c) in board.neighbors[(row, col)]:
        if board.grid[r][c] == value:
            return False
    return True

# AC3 revise: if cell2 is assigned (its domain is a single bit), remove that bit from cell1’s domain.
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
    # this should owrk because if dom2 is a power of 2, then it will only have one bit set to 1
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
            bool - True if the puzzle is solvable, False otherwise
    """
    queue = deque()
    for row in range(board.size):
        for col in range(board.size):
            for neighbor in board.neighbors[(row, col)]:
                queue.append(((row, col), neighbor))
    while queue:
        cell1, cell2 = queue.popleft()
        if revise(board, cell1, cell2):
            row, col = cell1
            if board.domains[row][col] == 0:
                return False
            for neighbor in board.neighbors[(row, col)]:
                if neighbor != cell2:
                    queue.append((neighbor, cell1))
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
    queue = deque([start_cell])
    while queue:
        r, c = queue.popleft()
        assigned_mask = board.domains[r][c]
        # Propagate the fact that (r,c) is assigned.
        for nr, nc in board.neighbors[(r, c)]:
            if board.domains[nr][nc] & assigned_mask:
                if (nr, nc) not in saved:
                    saved[(nr, nc)] = (board.domains[nr][nc], board.grid[nr][nc])
                board.domains[nr][nc] &= ~assigned_mask
                if board.domains[nr][nc] == 0:
                    return False, saved
                # If this neighbor becomes singleton and is not yet assigned, assign and propagate.
                if board.domains[nr][nc].bit_count() == 1 and board.grid[nr][nc] == 0:
                    val = board.domains[nr][nc].bit_length()  # For power-of-two, bit_length() gives the value.
                    board.grid[nr][nc] = val
                    queue.append((nr, nc))
    return True, saved

def restore_domains(board: SudokuBoard, saved: dict):
    """
        this method assists in restoring the last saved domains and grid values
        in essence, revert to the last state that was posible
    """
    for (r, c), (old_domain, old_grid) in saved.items():
        board.domains[r][c] = old_domain
        board.grid[r][c] = old_grid

def get_unassigned_variable(board: SudokuBoard):
    """
        this method gets the next unassigned variable in the grid using the minimum remaining vlaue
        heuristic--whcih cell has the fewest remaining values in its domain
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
    nbs = board.neighbors[(row, col)]
    def count_conflicts(val: int) -> int:
        bit = 1 << (val - 1)
        return sum(1 for (r, c) in nbs if board.domains[r][c] & bit)
    return sorted(values, key=count_conflicts)

# Backtracking search with MRV, LCV, and incremental propagation.
def backtrack(board: SudokuBoard) -> bool:
    """
        this method links together the usage of the mrv, lcv and incremental propogation methods.
        It uses a recursive backtracking algorithm by assigning values
        to the cells and checking if the assignment is valid. If it is, it will continue assigning
        values to the next cell and so on. If it reaches a point where it cannot assign a value to a cell
        it will backtrack and try a different value
    """
    if board.is_complete():
        return True
    var = get_unassigned_variable(board)
    if var is None:
        return False
    row, col = var
    sorted_vals = get_sorted_domain_values(board, row, col)
    orig_domain = board.domains[row][col]
    orig_grid_val = board.grid[row][col]
    for val in sorted_vals:
        if is_valid_assignment(board, row, col, val):
            board.assign_value(row, col, val)
            success, saved = incremental_propagate(board, (row, col))
            if success and backtrack(board):
                return True
            restore_domains(board, saved)
            board.domains[row][col] = orig_domain
            board.grid[row][col] = orig_grid_val
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
        this method writes the output of the sudoku puzzle to the console

        params:
            board: SudokuBoard - the board we are working with

        returns:
            None
    """
    if board.is_complete():
        board.display_board()
    else:
        print("No solution.")

def driver():
    grid = read_input()
    board = SudokuBoard(grid)
    if not ac3(board):
        print("No solution.")
        return
    if not backtrack(board):
        print("No solution.")
        return
    write_output(board)

if __name__ == '__main__':
    driver()