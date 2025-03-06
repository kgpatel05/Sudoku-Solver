"""
CSC 242 - Intro to AI
Project 2 - Sudoku
Christopher DelGuercio (cdelguer@u.rochester.edu)
Krish Patel (kpatel46@u.rochester.edu
"""


import sys
import numpy as np
from collections import deque

class SudokuBoard:
    def __init__(self, input_grid: list[list[int]]):
        self.size = 9
        self.subgrid_size = 3
        # Stores the puzzle as a Numpy array
        self.grid = np.array(input_grid, dtype=int)
        # Each cell gets a domain if empty or given number.
        self.domains = [
            [set(range(1, 10)) if self.grid[r, c] == 0 else {self.grid[r, c]}
            for c in range(self.size)]
            for r in range(self.size)
        ]

        # Precompute the neighbors for each cell
        self.neighbors = {}
        for r in range(self.size):
            for c in range(self.size):
                nbs = set()
                # All cells in the same row
                for cc in range(self.size):
                    if cc != c:
                        nbs.add((r, cc))
                # All cells in the same column
                for rr in range(self.size):
                    if rr != r:
                        nbs.add((rr, c))
                # All cells in the same subgrid
                br = (r // self.subgrid_size) * self.subgrid_size
                bc = (c // self.subgrid_size) * self.subgrid_size
                for rr in range(br, br + self.subgrid_size):
                    for cc in range(bc, bc + self.subgrid_size):
                        if (rr, cc) != (r, c):
                            nbs.add((rr, cc))
                self.neighbors[(r, c)] = tuple(nbs)

    def is_complete(self):
        return not np.any(self.grid == 0)

    def assign_value(self, row: int, col: int, value: int):
        self.grid[row, col] = value
        self.domains[row][col] = {value}

    def remove_value(self, row: int, col: int):
        self.grid[row, col] = 0
        self.domains[row][col] = set(range(1, 10))

    def display_board(self):
        for r in range(self.size):
            print(" ".join(str(x) for x in self.grid[r]))


# CONSTRAINT CHECKING
# -------------------
# Check if value assignment is valid
def is_valid_assignment(board: SudokuBoard, row: int, col: int, value: int) -> bool:
    if np.any(board.grid[row, :] == value):
        return False
    if np.any(board.grid[:, col] == value):
        return False
    br = (row // board.subgrid_size) * board.subgrid_size
    bc = (col // board.subgrid_size) * board.subgrid_size
    if np.any(board.grid[br:br+board.subgrid_size, bc:bc+board.subgrid_size] == value):
        return False
    return True

# Remove from cell1 domain any value that cell2 already has
def revise(board: SudokuBoard, cell1: tuple[int, int], cell2: tuple[int, int]) -> bool:
    r1, c1 = cell1
    r2, c2 = cell2
    revised = False
    if len(board.domains[r2][c2]) == 1:
        val = next(iter(board.domains[r2][c2]))
        if val in board.domains[r1][c1]:
            board.domains[r1][c1].remove(val)
            revised = True
    return revised


# MAIN ALGORITHM
# --------------
# use AC3 to prune domains by making sure every value is consistent with neighbors
def ac3(board: SudokuBoard) -> bool:
    queue = deque()
    for r in range(board.size):
        for c in range(board.size):
            for neighbor in board.neighbors[(r, c)]:
                queue.append(((r, c), neighbor))
    while queue:
        cell1, cell2 = queue.popleft()
        if revise(board, cell1, cell2):
            r, c = cell1
            if not board.domains[r][c]:
                return False
            for neighbor in board.neighbors[(r, c)]:
                if neighbor != cell2:
                    queue.append((neighbor, cell1))
    return True


# HEURISTICS/BACKTRACKING SEARCH
# ------------------------------
# Use MRV to pick the next cell (smallest domain)  **heuristic
def get_unassigned_variable(board: SudokuBoard):
    min_len = 10
    chosen = None
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r, c] == 0:
                d_len = len(board.domains[r][c])
                if d_len < min_len:
                    min_len = d_len
                    chosen = (r, c)
    return chosen

# Order possible values for a cell using LCV  **heuristic
def get_sorted_domain_values(board: SudokuBoard, row: int, col: int) -> list[int]:
    values = list(board.domains[row][col])
    nbs = board.neighbors[(row, col)]
    def count_conflicts(val: int) -> int:
        cnt = 0
        for (r, c) in nbs:
            if val in board.domains[r][c]:
                cnt += 1
        return cnt
    return sorted(values, key=count_conflicts)

# After assigning value only update the neighbors that might be affected... **runtime optimization
def incremental_propagate(board: SudokuBoard, start_cell: tuple[int, int]) -> (bool, dict):
    removed = {}
    queue = deque()
    for neighbor in board.neighbors[start_cell]:
        queue.append((neighbor, start_cell))
    while queue:
        cell1, cell2 = queue.popleft()
        r2, c2 = cell2
        if len(board.domains[r2][c2]) == 1:
            val = next(iter(board.domains[r2][c2]))
            r1, c1 = cell1
            if val in board.domains[r1][c1]:
                board.domains[r1][c1].remove(val)
                removed.setdefault(cell1, set()).add(val)
                if len(board.domains[r1][c1]) == 0:
                    return False, removed
                for neighbor in board.neighbors[cell1]:
                    if neighbor != cell2:
                        queue.append((neighbor, cell1))
    return True, removed

# If a branch fails revert changes
def restore_domains(board: SudokuBoard, removed: dict):
    for (r, c), vals in removed.items():
        board.domains[r][c].update(vals)

# Backtracking search with MRV, LCV, and incremental propagation
def backtrack(board: SudokuBoard) -> bool:
    if board.is_complete():
        return True
    var = get_unassigned_variable(board)
    if var is None:
        return False
    row, col = var
    sorted_vals = get_sorted_domain_values(board, row, col)
    for val in sorted_vals:
        if is_valid_assignment(board, row, col, val):
            orig_domain = board.domains[row][col].copy()
            board.assign_value(row, col, val)
            success, removed = incremental_propagate(board, (row, col))
            if success and backtrack(board):
                return True
            restore_domains(board, removed)
            board.domains[row][col] = orig_domain
            board.grid[row, col] = 0
    return False


# INPUT/OUTPUT HANDLING
# ---------------------
def read_input() -> list[list[int]]:
    grid = []
    for _ in range(9):
        line = sys.stdin.readline().strip()
        if not line:
            break
        grid.append(list(map(int, line.split())))
    return grid

def write_output(board: SudokuBoard) -> None:
    if board.is_complete():
        board.display_board()
    else:
        print("No solution.")

def main():
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
    main()