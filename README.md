# Python CP Setup

This repository offers a comprehensive setup for competitive programming in Python. It includes VS Code configurations, scripts, boilerplate templates, commonly used data structures and algorithms, and a lot more, ensuring an efficient coding environment for CPers of all levels.

## Setup

### PyPy Setup

1. **Download PyPy**  

   You can download PyPy from the official website: [https://pypy.org/download.html](https://pypy.org/download.html)

3. **Setup Virtual Environment with Poetry**  

   After downloading PyPy, you can set it up in a virtual environment using Poetry with the following command:
   ```bash
   poetry env use /path/to/pypy3.10/bin/pypy3
   ```

5. **Configure in VS Code**
   - Select the Python interpreter in VS Code and set it to the PyPy interpreter located at `.venv/bin/pypy3`.
   - Update the `cph.language.python.Command` in the `.vscode/settings.json` file to point to the absolute path of the PyPy interpreter inside your virtual environment.

### VS Code Extensions Setup

Extensions that should be added to VS Code:

- **[Competitive Programming Helper (cph)](https://marketplace.visualstudio.com/items?itemName=DivyanshuAgrawal.competitive-programming-helper)**: Simplifies the process of downloading problems, compiling, and judging in competitive programming.
- **[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)**: Provides essential Python language support, including IntelliSense, debugging, and linting.
- **[Task Explorer](https://marketplace.visualstudio.com/items?itemName=spmeesseman.vscode-taskexplorer)**: Manages tasks like running code or scripts, so we don't have to write commands every time.

### Setup for type_from_clipboard.bash

1. **Make Script Executable**
   Run the following command to make the script executable:
   ```bash
   chmod +x scripts/type_from_clipboard.bash
   ```

2. **Install Required Tools**
   Install the necessary tools for the script:
   ```bash
   sudo apt-get install xclip xdotool
   ```

## Usage

### main.py

This file includes commonly used library imports, boilerplate code with functions, decorators, and options for quick maneuvering.

- `solve_brute_force` function: Implements a brute force solution to compare against the efficient solution using random test cases generated by the `test_with_random_cases.py` script.
- `execute_once` function: Runs once at the beginning to precompute values for all test cases, assigning them to global variables.
- `solve` function: Called for each test case to run the actual solution.
- `INPUT_NUMBER_OF_TEST_CASES`: If enabled, the program reads the number of test cases; otherwise, it defaults to running a single test case.
- `BOOLEAN_RETURN`: If set, allows the function to return `True`/`False` instead of printing, with the output customizable using `TRUE_MAPPING` and `FALSE_MAPPING` for `YES`/`NO` or other text.

There are two ways to debug it:

1. VS Code Debugger: Provides more control over debugging, allowing you to set breakpoints and step through code. Uses `debug/input.txt` and `debug/output.txt` for input and output respectively.
2. Debug Function: Prints debug information to `stderr` if `DEBUG_ENABLED` is set.

### addon.py

This file contains commonly used algorithms and data structures that can be helpful for specific problems.

### maths.py

This file contains classes with utility functions for combinatorics, factorial, and prime number-related operations.

### hash.py

This file contains classes with utility functions for rolling and range hashing of strings.

### scripts/test_with_random_cases.py

- `test_case_generator` function: Generates random test cases, managing the input for each test case. Here's an example of generating a test case with size of array as the first input and an integer array as the second input:
   ```python
   def test_case_generator(rig: RandomInputGenerator, im: InputManager):
      n = rig.random_int(1, 10)
      arr = rig.random_array(n, 50, 100)
      im.add(n)
      im.add(arr)
   ```
- `NUMBER_OF_TEST_CASES`: Set how many test cases you want to generate and test.

### scripts/type_from_clipboard.bash

This script types text from the clipboard, useful when the platform doesn't support copy/paste outside of an integrated IDE. When executed, it starts typing from the clipboard after a 5-second delay.

## Acknowledgements

The data structures and algorithms in this repository are primarily sourced from other people's submissions on competitive programming platforms. These implementations have been modified to make them more beginner-friendly. In particular, many of the algorithms originate from repositories such as [PyRival](https://github.com/cheran-senthil/PyRival) and [ac-library-python](https://github.com/not522/ac-library-python). For additional resources not included in this setup, visit these repositories.
