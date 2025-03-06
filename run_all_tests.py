
import subprocess
import glob
import os
import time


def run_test(test_file: str) -> str:
    """Run main.py with the test file as input and return the program output."""
    with open(test_file, "r") as f:
        input_data = f.read()
    result = subprocess.run(
        ["python3", "main.py"],
        input=input_data,
        text=True,
        capture_output=True
    )
    return result.stdout


def main():
    start_time = time.time()

    # Gather all test files from "easy" and "hard" directories.
    test_files = glob.glob(os.path.join("easy", "*.txt")) + glob.glob(os.path.join("hard", "*.txt"))
    test_files.sort()  # Optional: sort for a consistent order.

    for test_file in test_files:
        print(f"Running test: {test_file}")
        output = run_test(test_file)
        print("Output:")
        print(output)
        print("-" * 40)

    total_time = time.time() - start_time
    print(f"Total time to run all tests: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
