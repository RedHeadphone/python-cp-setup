import random, sys
from importlib import reload
from io import StringIO
from unittest.mock import patch

def mock_input(inputs):
    def generator():
        for item in inputs:
            yield item
    gen = generator()
    return lambda: next(gen)

def test_solve(inputs, function):
    with patch('sys.stdin.readline', new=mock_input(inputs)):
        captured_output = StringIO()
        sys.stdout = captured_output
        function_output = function()
        sys.stdout = sys.__stdout__
        console_output = captured_output.getvalue()

        return function_output, console_output

class RandomInputGenerator:
    def random_int(self,lower = 5, upper = 20):
        return random.randint(lower, upper)

    def random_array(self,size = 10, lower = 0, upper = 100):
        return [self.random_int(lower, upper) for _ in range(size)]

    def randome_binary_string(self,size = 10):
        return ''.join([random.choice('01') for _ in range(size)])

class InputManager:
    def __init__(self):
        self.inputs = []
    
    def serialize(self,input):
        if type(input) == list:
            return " ".join([str(i) for i in input])
        return str(input)

    def add(self, input):
        self.inputs.append(self.serialize(input))


###############################################################################

def test_case_generator(rig: RandomInputGenerator, im: InputManager):
    pass

NUMBER_OF_TEST_CASES = 20
COMPARE_FUNCTION_OUTPUT = 1
COMPARE_CONSOLE_OUTPUT = 1

###############################################################################


test_cases = []
for t in range(NUMBER_OF_TEST_CASES):
    rig = RandomInputGenerator()
    im = InputManager()
    test_case_generator(rig, im)
    test_cases.append(im.inputs)

while True:
    import main
    reload(main)

    failed_inputs = []

    for inp in test_cases:
        bfo,bco = test_solve(inp, main.solve_bruteforce)
        fo,co = test_solve(inp, main.solve)
        if (COMPARE_FUNCTION_OUTPUT and bfo != fo) or (COMPARE_CONSOLE_OUTPUT and bco != co):
            failed_inputs.append((inp,bfo,fo,bco,co))

    if len(failed_inputs) != 0:
        print(f"{len(failed_inputs)}/{NUMBER_OF_TEST_CASES} failed!")
        with open("test/failed_cases.txt", "w") as f:
            for inp,bfo,fo,bco,co in failed_inputs:
                f.write("input: \n" + '\n'.join(inp) + "\n")
                f.write("bruteforce output: \n" + str(bfo) + "\n")
                f.write("output: \n" + str(fo) + "\n")
                f.write("bruteforce console output: \n" + str(bco))
                f.write("console output: \n" + str(co))
                f.write("\n\n")
        if input("Try again? (Y/n) ").lower() == "n":
            break
    else:
        with open("test/failed_cases.txt", "w") as f:
            f.write("")
        print("All tests passed!")
        break
