#!/usr/bin/env python3
"""
Name: Your Name
Email: your.email@example.com

CSC242: Intro to AI – Project 2: Constraint Satisfaction (Sudoku Solver)

This program reads a 9x9 Sudoku puzzle from standard input (with 0 representing an empty cell),
applies AC-3 constraint propagation followed by backtracking search with MRV and LCV heuristics,
and then outputs a solved puzzle (or "No solution." if none exists).
"""

import sys
import copy
from collections import deque
import numpy as np


class SudokuBoard:
    def __init__(self, input_grid: list[list[int]]):
        self.size: int = 9
        self.subgrid_size: int = 3
        # Make a copy of the input grid
        self.grid: list[list[int]] = [row[:] for row in input_grid]
        # Initialize domains: if cell is 0, its domain is {1,...,9}; otherwise, it's the singleton of its value.
        self.domains: list[list[set[int]]] = [
            [set(range(1, 10)) if self.grid[r][c] == 0 else {self.grid[r][c]}
             for c in range(self.size)]
            for r in range(self.size)
        ]
        # Precompute neighbors for each cell (same row, same column, same subgrid)
        self.neighbors: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for r in range(self.size):
            for c in range(self.size):
                nbs = set()
                # Same row
                for cc in range(self.size):
                    if cc != c:
                        nbs.add((r, cc))
                # Same column
                for rr in range(self.size):
                    if rr != r:
                        nbs.add((rr, c))
                # Same subgrid
                br = (r // self.subgrid_size) * self.subgrid_size
                bc = (c // self.subgrid_size) * self.subgrid_size
                for rr in range(br, br + self.subgrid_size):
                    for cc in range(bc, bc + self.subgrid_size):
                        if (rr, cc) != (r, c):
                            nbs.add((rr, cc))
                self.neighbors[(r, c)] = nbs

    def is_complete(self) -> bool:
        """A board is complete if every cell is assigned (i.e. grid value nonzero) and its domain is a singleton."""
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == 0 or len(self.domains[r][c]) != 1:
                    return False
        return True

    def is_valid_assignment(self, row: int, col: int, value: int) -> bool:
        """Checks whether placing 'value' at (row, col) conflicts with any assigned value in the same row, column, or subgrid."""
        # Check row
        for c in range(self.size):
            if self.grid[row][c] == value:
                return False
        # Check column
        for r in range(self.size):
            if self.grid[r][col] == value:
                return False
        # Check subgrid
        br = (row // self.subgrid_size) * self.subgrid_size
        bc = (col // self.subgrid_size) * self.subgrid_size
        for r in range(br, br + self.subgrid_size):
            for c in range(bc, bc + self.subgrid_size):
                if self.grid[r][c] == value:
                    return False
        return True

    def assign_value(self, row: int, col: int, value: int) -> None:
        """Assigns 'value' to cell (row, col) and updates its domain."""
        self.grid[row][col] = value
        self.domains[row][col] = {value}

    def remove_value(self, row: int, col: int) -> None:
        """Removes the assignment at cell (row, col) and resets its domain."""
        self.grid[row][col] = 0
        self.domains[row][col] = set(range(1, 10))

    def display_board(self) -> None:
        """Prints the board in the required 9-line, 9-token-per-line format."""
        for row in self.grid:
            print(" ".join(str(x) for x in row))


def revise(board: SudokuBoard, cell1: tuple[int, int], cell2: tuple[int, int]) -> bool:
    """Revise the domain of cell1 with respect to cell2.
    Remove any value from cell1's domain if cell2 is assigned that value.
    """
    r1, c1 = cell1
    r2, c2 = cell2
    revised = False
    to_remove = set()
    # If cell2's domain is a singleton, then its value cannot appear in cell1.
    if len(board.domains[r2][c2]) == 1:
        value = next(iter(board.domains[r2][c2]))
        for v in board.domains[r1][c1]:
            if v == value:
                to_remove.add(v)
    if to_remove:
        board.domains[r1][c1] -= to_remove
        revised = True
    return revised


def ac3(board: SudokuBoard) -> bool:
    """Applies the AC-3 algorithm to enforce arc consistency across the board.
    Returns False if any domain becomes empty, True otherwise.
    """
    queue = deque()
    # Initialize the queue with all arcs: (cell, neighbor)
    for r in range(board.size):
        for c in range(board.size):
            for neighbor in board.neighbors[(r, c)]:
                queue.append(((r, c), neighbor))
    while queue:
        (r, c), (r2, c2) = queue.popleft()
        if revise(board, (r, c), (r2, c2)):
            if not board.domains[r][c]:
                return False
            for neighbor in board.neighbors[(r, c)]:
                if neighbor != (r2, c2):
                    queue.append((neighbor, (r, c)))
    return True


def get_unassigned_variable(board: SudokuBoard) -> tuple[int, int] | None:
    """Selects an unassigned cell using the Minimum Remaining Values (MRV) heuristic."""
    min_len = 10  # larger than the maximum domain size of 9
    chosen = None
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r][c] == 0:
                d_len = len(board.domains[r][c])
                if d_len < min_len:
                    min_len = d_len
                    chosen = (r, c)
    return chosen


def get_sorted_domain_values(board: SudokuBoard, row: int, col: int) -> list[int]:
    """Returns the domain values for (row, col) ordered by the Least Constraining Value (LCV) heuristic."""
    values = list(board.domains[row][col])

    def count_conflicts(value: int) -> int:
        count = 0
        for (r, c) in board.neighbors[(row, col)]:
            if value in board.domains[r][c]:
                count += 1
        return count

    return sorted(values, key=count_conflicts)


def backtrack(board: SudokuBoard) -> bool:
    """Performs backtracking search to complete the board.
    Returns True if a complete solution is found; otherwise, returns False.
    """
    if board.is_complete():
        return True

    var = get_unassigned_variable(board)
    if var is None:
        return False
    row, col = var

    for value in get_sorted_domain_values(board, row, col):
        if board.is_valid_assignment(row, col, value):
            # Save the current state so we can backtrack later.
            old_grid = copy.deepcopy(board.grid)
            old_domains = copy.deepcopy(board.domains)

            board.assign_value(row, col, value)
            # Inference: enforce AC-3 after the assignment.
            if ac3(board):
                if backtrack(board):
                    return True
            # Backtrack: revert to the previous state.
            board.grid = old_grid
            board.domains = old_domains
    return False


def read_input() -> list[list[int]]:
    """Reads a 9x9 Sudoku puzzle from standard input.
    Each of the 9 lines should contain 9 digits (0-9) separated by spaces.
    """
    grid = []
    for _ in range(9):
        line = sys.stdin.readline().strip()
        if not line:
            break
        row = list(map(int, line.split()))
        grid.append(row)
    return grid


def write_output(board: SudokuBoard) -> None:
    """Outputs the solved board or 'No solution.' if the board is incomplete."""
    if board.is_complete():
        board.display_board()
    else:
        print("No solution.")


def main():
    """
    General Pipeline:
      1. Read input. (read_input())
      2. Initialize board. (SudokuBoard)
      3. Apply AC-3. (ac3(board))
      4. Use backtracking search. (backtrack(board))
      5. Output solution. (write_output(board))
    """
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