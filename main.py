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


    def revise(self, row1: int, col1: int, row2: int, col2: int) -> bool:
        """
        Removes values from (row1, col1) that do not satisfy constraints with (row2, col2).
        Returns True if domain is revised, False otherwise.
        """
        revised = False
        to_remove = set()

        for value in self.domains[row1][col1]:
            # If (row2, col2) has only one value and it's in (row1, col1), remove it
            if len(self.domains[row2][col2]) == 1 and value in self.domains[row2][col2]:
                to_remove.add(value)

        if to_remove:
            self.domains[row1][col1] -= to_remove  # Remove invalid values
            revised = True

        return revised


    def _generate_constraints(self) -> set[tuple[int, int]]:
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
        """
            This method checks if the proposed assignment (from the params)
            is a valid assignment for the given position.

            params:
                row: int - The row of the cell to check
                col: int - The column of the cell to check
                value: int - The value to check

            returns:
                bool - True if the assignment is valid, False otherwise
        """
        ##check if row assignment is valid
        if value in self.grid[row]:
            return False
        ##check if column assignment is valid
        if value in self.grid[:, col]:
            return False

        ##check if subgrid assignment is valid
        start_row = (row // self.subgrid_size) * self.subgrid_size
        start_col = (col // self.subgrid_size) * self.subgrid_size
        if value in self.grid[start_row:start_row+self.subgrid_size, start_col:start_col+self.subgrid_size]:
            return False
        
        return True

    def assign_value(self, row: int, col: int, value: int) -> None:
        """
            Assigns a value to a given cell in the grid.

            params:
                row: int - The row of the cell to assign
                col: int - The column of the cell to assign
                value: int - The value to assign

            returns:
                None
        """

        # update value on the grid 
        self.grid[row][col] = value

        # update domain of the cell
        self.domains[row][col] = {value}

        # remove value from 'peer' cells
        for (r, c) in self._get_peers(row, col):
            self.domains[r][c].discard(value)

        return
    
    def _get_peers(self, row: int, col: int) -> set[tuple[int, int]]:
        """
            This methods all boxes that are in the same row, column and grid of the
            specified position. This in essence is collecting all of the values that
            can restrict the value of the specified position.
        """
        peers = set()

        # Row and Column peers
        for i in range(self.size):
            if i != col:
                peers.add((row, i))
            if i != row:
                peers.add((i, col))

        # Subgrid peers
        start_row, start_col = (row // self.subgrid_size) * self.subgrid_size, (col // self.subgrid_size) * self.subgrid_size
        for r in range(start_row, start_row + self.subgrid_size):
            for c in range(start_col, start_col + self.subgrid_size):
                if (r, c) != (row, col):
                    peers.add((r, c))

        return peers

    def remove_value(self, row: int, col: int) -> None:
        """
            This method restores teh value of the specified cell to
            0 and restores the domain of that position as well.

            params:
                row: int - The row of the cell to remove
                col: int - The column of the cell to remove

            returns:
                None
        """
        self.grid[row, col] = 0 

        # Restore possible values (1-9) minus currently assigned numbers in peers
        possible_values = set(range(1, 10))
        for (r, c) in self._get_peers(row, col):
            possible_values.discard(self.grid[r, c])
        
        self.domains[row][col] = possible_values

        return
    
    def is_valid_board(self) -> bool:
        """
        HELPER METHOD FOR TESTING ONLY
        Checks if the current Sudoku board is valid (i.e., no duplicate values in rows, columns, or subgrids).

        Returns:
            bool: True if the board is valid, False otherwise.
        """
        # Check rows
        for row in range(self.size):
            row_values = [self.grid[row, col] for col in range(self.size) if self.grid[row, col] != 0]
            if len(set(row_values)) != len(row_values):  # Duplicates found
                return False

        # Check columns
        for col in range(self.size):
            col_values = [self.grid[row, col] for row in range(self.size) if self.grid[row, col] != 0]
            if len(set(col_values)) != len(col_values):  # Duplicates found
                return False

        # Check 3x3 subgrids
        for start_row in range(0, self.size, self.subgrid_size):
            for start_col in range(0, self.size, self.subgrid_size):
                subgrid_values = []
                for r in range(start_row, start_row + self.subgrid_size):
                    for c in range(start_col, start_col + self.subgrid_size):
                        if self.grid[r, c] != 0:
                            subgrid_values.append(self.grid[r, c])
                if len(set(subgrid_values)) != len(subgrid_values):  # Duplicates found
                    return False

        return True  # If all checks pass, the board is valid


def ac3(board: SudokuBoard) -> bool:
    """
        TODO: write a description of this method
    """
    queue = list(board.constraints)

    while queue:
        (cell1, cell2) = queue.pop(0)
        row1, col1 = cell1
        row2, col2 = cell2

        if board.revise(row1, col1, row2, col2):
            if not board.domains[row1][col1]:  # If domain is empty, failure
                return False
            for peer in board._get_peers(row1, col1):
                queue.append((peer, (row1, col1)))

    return True


def revise(self, row1: int, col1: int, row2: int, col2: int) -> bool:
    """
        # TODO: write a description of this method
    """
    revised = False
    to_remove = set()

    for value in self.domains[row1][col1]:
        if len(self.domains[row2][col2]) == 1 and value in self.domains[row2][col2]:
            to_remove.add(value)

    if to_remove:
        self.domains[row1][col1] -= to_remove
        revised = True

    return revised


def get_unassigned_variable(board: SudokuBoard) -> tuple[int, int] | None:
    """
        TODO: write a description of this method

        uses the minimum remaining values heuristic
    """
    min_domain_size = float('inf')
    best_cell = None

    for row in range(board.size):
        for col in range(board.size):
            if board.grid[row, col] == 0:  # Unassigned cell
                domain_size = len(board.domains[row][col])
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    best_cell = (row, col)

    return best_cell


def get_sorted_domain_values(board: SudokuBoard, row: int, col: int) -> list[int]:
    """
        TODO: write a description of this method
        Returns a list of values in the domain of (row, col), sorted by the Least Constraining Value (LCV) heuristic.
        The LCV heuristic prefers values that minimize constraints on neighboring cells.
    """
    def _count_conflicts(value: int) -> int:
        """
        TODO: write a description of this method
            Helper function that counts how many values would be removed from the domains
            of peers if we assigned 'value' to (row, col).
        """
        conflicts = 0
        for (r, c) in board._get_peers(row, col):
            if value in board.domains[r][c]:
                conflicts += 1
        return conflicts

    # Get all possible values and sort them based on the number of conflicts they cause
    return sorted(board.domains[row][col], key=_count_conflicts)


def backtrack(board: SudokuBoard) -> bool:
    """
        TODO: write a description of this method
    """
    if board.is_complete():
        return True  # Solved

    row, col = get_unassigned_variable(board)
    
    # Use LCV heuristic
    for value in get_sorted_domain_values(board, row, col):
        if board.is_valid_assignment(row, col, value):
            board.assign_value(row, col, value)

            if backtrack(board):  # Recursion
                return True  # Success

            board.remove_value(row, col)  # Undo if failure

    return False  # No solution found


def read_input() -> list[list[int]]:
    """
    Reads a 9x9 Sudoku grid from standard input using the built-in input() function.

    Returns:
        A list of lists (9x9) containing the Sudoku puzzle, where 0 represents an empty cell.
    """
    input_grid = []
    input("Press Enter to start entering the Sudoku puzzle...")

    for _ in range(9):  # Read exactly 9 lines
        line = input().strip()  # Explicit prompt for clarity
        input_grid.append(list(map(int, line.split())))  # Convert to list of integers
    
    return input_grid


def write_output(board: SudokuBoard) -> None:
    """
        TODO: write a description of this method
    """
    print(board.is_valid_board())
    if board.is_complete():
        for row in board.grid:
            print(" ".join(map(str, row)))  # Convert row to space-separated string
    else:
        print("No solution.")

def main():
    """
    General Pipeline:
    - Read input. (read_input())
    - Initialize board. (SudokuBoard)
    - Apply AC-3. (ac3(board))
    - Use backtracking search. (backtrack(board))
    - Output solution. (write_output(board.grid))
    """
    # Step 1: Read the puzzle input
    input_grid = read_input()

    # Step 2: Initialize the Sudoku board
    board = SudokuBoard(input_grid)

    # Step 3: Apply AC-3 to reduce domains
    ac3(board)

    # Step 4: If AC-3 didn't solve the puzzle, use backtracking
    if not board.is_complete():
        backtrack(board)

    # Step 5: Output the result
    write_output(board)


main()