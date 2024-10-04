import random, sys
from os import path
from importlib import reload
from io import StringIO
from unittest.mock import patch

sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))


def mock_input(inputs):
    def generator():
        for item in inputs:
            yield item

    gen = generator()
    return lambda: next(gen)


def test_solve(inputs, function):
    with patch("sys.stdin.readline", new=mock_input(inputs)):
        captured_output = StringIO()
        sys.stdout = captured_output
        function_output = function()
        sys.stdout = sys.__stdout__
        console_output = captured_output.getvalue()

        return function_output, console_output


class RandomInputGenerator:
    def random_int(self, lower=5, upper=20):
        return random.randint(lower, upper)

    def random_array(self, size=10, lower=0, upper=100):
        return [self.random_int(lower, upper) for _ in range(size)]

    def randome_binary_string(self, size=10):
        return "".join([random.choice("01") for _ in range(size)])


class InputManager:
    def __init__(self):
        self.inputs = []

    def serialize(self, input):
        if type(input) == list:
            return " ".join([str(i) for i in input])
        return str(input)

    def add(self, input):
        self.inputs.append(self.serialize(input))


###############################################################################


def test_case_generator(rig: RandomInputGenerator, im: InputManager):
    pass


NUMBER_OF_TEST_CASES = 20

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
        bfo, bco = test_solve(inp, main.solve_bruteforce)
        fo, co = test_solve(inp, main.solve)
        if bfo != fo or bco != co:
            failed_inputs.append((inp, bfo, fo, bco, co))

    if len(failed_inputs) != 0:
        print(f"{len(failed_inputs)}/{NUMBER_OF_TEST_CASES} failed!\n")
        for inp, bfo, fo, bco, co in failed_inputs:
            print("input: \n" + "\n".join(inp))
            if bfo != fo:
                print("function output: \n" + str(fo))
                print("bruteforce function output: \n" + str(bfo))
            if bco != co:
                print("console output: \n" + str(co), end="")
                print("bruteforce console output: \n" + str(bco))
        if input("Try again? (Y/n) ").lower() == "n":
            break
    else:
        print("All tests passed!")
        break
