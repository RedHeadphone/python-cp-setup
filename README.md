# Python CP Setup

This repository offers a comprehensive setup for competitive programming in Python. It includes VS Code configurations, scripts, boilerplate templates, commonly used data structures and algorithms, and a lot more, ensuring an efficient coding environment for CPers of all levels.

## Setup

### PyPy Setup

1. **Download PyPy**  
   You can download PyPy from the official website: [https://pypy.org/download.html](https://pypy.org/download.html)

2. **Setup Virtual Environment with Poetry**  
   After downloading PyPy, you can set it up in a virtual environment using Poetry with the following command:
   ```bash
   poetry env use /path/to/pypy3.10/bin/pypy3
   ```

3. **Configure in VS Code**

   - Select the Python interpreter in VS Code and set it to the PyPy interpreter located at `.venv/bin/pypy3`.
   - Update the `cph.language.python.Command` in the `.vscode/settings.json` file to point to the absolute path of the PyPy interpreter inside your virtual environment.

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

### VS Code Extensions Setup

Extensions that should be added to VS Code:

- **[Competitive Programming Helper (cph)](https://marketplace.visualstudio.com/items?itemName=DivyanshuAgrawal.competitive-programming-helper)**: Simplifies the process of downloading problems, compiling, and judging in competitive programming.

- **[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)**: Provides essential Python language support, including IntelliSense, debugging, and linting.

- **[Task Explorer](https://marketplace.visualstudio.com/items?itemName=spmeesseman.vscode-taskexplorer)**: Manages tasks like running code or scripts, so we don't have to write commands every time.

## Usage

### main.py

### addon.py

### scripts/test_with_random_cases.py

### scripts/type_from_clipboard.bash

## Acknowledgements