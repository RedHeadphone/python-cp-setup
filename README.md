# Python CP Setup

This repository offers a comprehensive setup for competitive programming in Python. It includes VS Code configurations, scripts, boilerplate templates, commonly used data structures and algorithms, and a lot more, ensuring an efficient coding environment for CPers of all levels.

### PyPy Setup

1. **Download PyPy**  
   You can download PyPy from the official website: [https://pypy.org/download.html](https://pypy.org/download.html)

2. **Setup Virtual Environment with Poetry**  
   After downloading PyPy, you can set it up in a virtual environment using Poetry with the following command:
   ```bash
   poetry env use /path/to/pypy3.10/bin/pypy3
   ```

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