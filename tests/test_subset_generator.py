import unittest
from .utils import describe_test
from information_noise_reduction.subset_generator import all_subsets_generator, reverse_all_subsets_generator

class TestSubsetGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        describe_test("SubsetGenerator")

    def run(self, test):
        result = super().run(test)
        print("\n" + "=" * 30 + "\n")
        return result
    
    def test_all_subset(self):
        columns = ["A", "B", "C"]
        expected_subsets = {
            frozenset(["A"]), frozenset(["B"]), frozenset(["C"]),
            frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"]),
            frozenset(["A", "B", "C"])
        }
        result = {frozenset(subset) for subset in all_subsets_generator(columns)}
        self.assertEqual(result, expected_subsets)

    def test_reverse_all_subsets(self):
        columns = ["A", "B", "C"]
        expected_subsets = [
            frozenset(["A", "B", "C"]),
            frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"]),
            frozenset(["A"]), frozenset(["B"]), frozenset(["C"])
        ]
        result = [frozenset(subset) for subset in reverse_all_subsets_generator(columns)]
        self.assertEqual(result, expected_subsets)
