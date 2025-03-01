import sys
import numpy as np

class SudokuBoard:
    def __init__(self, input_grid: list[list[int]]):
        self.size = 9
        self.subgrid_size = 3
        # Use a NumPy array for the grid (for fast row/col/subgrid checks)
        self.grid = np.array(input_grid, dtype=int)
        # Domains are kept as a list-of-lists of sets (one per cell)
        self.domains = [[set(range(1, 10)) if self.grid[r, c] == 0 else {self.grid[r, c]}
                         for c in range(self.size)] for r in range(self.size)]
        # Precompute neighbors for each cell (cells sharing a row, column, or subgrid)
        self.neighbors = {}
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

    def is_complete(self):
        # With a NumPy array, we can quickly check for any zeros.
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


def is_valid_assignment(board: SudokuBoard, row: int, col: int, value: int) -> bool:
    # Check row, column, and subgrid using NumPy operations
    if np.any(board.grid[row, :] == value):
        return False
    if np.any(board.grid[:, col] == value):
        return False
    br = (row // board.subgrid_size) * board.subgrid_size
    bc = (col // board.subgrid_size) * board.subgrid_size
    if np.any(board.grid[br:br+board.subgrid_size, bc:bc+board.subgrid_size] == value):
        return False
    return True

def revise(board: SudokuBoard, cell1: tuple[int, int], cell2: tuple[int, int]) -> bool:
    r1, c1 = cell1
    r2, c2 = cell2
    revised = False
    # If cell2 is already assigned, remove that value from cell1's domain.
    if len(board.domains[r2][c2]) == 1:
        val = next(iter(board.domains[r2][c2]))
        if val in board.domains[r1][c1]:
            board.domains[r1][c1].remove(val)
            revised = True
    return revised

def ac3(board: SudokuBoard) -> bool:
    # Build a simple queue (using a list) of all arcs.
    queue = []
    for r in range(board.size):
        for c in range(board.size):
            for neighbor in board.neighbors[(r, c)]:
                queue.append(((r, c), neighbor))
    while queue:
        cell1, cell2 = queue.pop(0)
        if revise(board, cell1, cell2):
            r, c = cell1
            if not board.domains[r][c]:
                return False
            for neighbor in board.neighbors[(r, c)]:
                if neighbor != cell2:
                    queue.append((neighbor, cell1))
    return True

def get_unassigned_variable(board: SudokuBoard):
    # MRV: choose the unassigned cell with the smallest domain size.
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

def get_sorted_domain_values(board: SudokuBoard, row: int, col: int) -> list[int]:
    # LCV: order values by the number of conflicts they cause
    values = list(board.domains[row][col])
    def count_conflicts(val: int) -> int:
        cnt = 0
        for (r, c) in board.neighbors[(row, col)]:
            if val in board.domains[r][c]:
                cnt += 1
        return cnt
    return sorted(values, key=count_conflicts)

def forward_check(board: SudokuBoard, row: int, col: int, value: int):
    # Remove 'value' from the domains of all neighbors;
    # record all removals in a dictionary so they can be undone.
    removed = {}
    for (r, c) in board.neighbors[(row, col)]:
        if value in board.domains[r][c]:
            board.domains[r][c].remove(value)
            if (r, c) in removed:
                removed[(r, c)].add(value)
            else:
                removed[(r, c)] = {value}
            if len(board.domains[r][c]) == 0:
                return False, removed
    return True, removed

def restore_domains(board: SudokuBoard, removed: dict):
    # Undo the removals made during forward checking.
    for (r, c), vals in removed.items():
        board.domains[r][c].update(vals)

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
            # Save a copy of the current domain for this cell.
            orig_domain = board.domains[row][col].copy()
            board.assign_value(row, col, val)
            success, removed = forward_check(board, row, col, val)
            if success and backtrack(board):
                return True
            # Undo forward checking and the assignment.
            restore_domains(board, removed)
            board.domains[row][col] = orig_domain
            board.grid[row, col] = 0
    return False

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
    # start_time = time.time()
    board = SudokuBoard(grid)
    # Run AC-3 once at the start
    if not ac3(board):
        print("No solution.")
        return
    if not backtrack(board):
        print("No solution.")
        return
    # end_time = time.time()
    write_output(board)
    # print("Time:", end_time - start_time)

if __name__ == '__main__':
    main()
