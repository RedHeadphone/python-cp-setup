# CP-template-python

Python Template for Competitive Programming. Open to Pull Request !

Running main.py to get output of input.txt
```bash
python main.py < input.txt > output.txt
```

Generating inputs to input.txt
```bash
python testcase_gen.py > input.txt
```

Comparing outputs of bruteforce solution and your solution
```bash
python main.py < input.txt > output.txt && python bruteforce.py < input.txt > output_bruteforce.txt

diff output.txt output_bruteforce.txt
```