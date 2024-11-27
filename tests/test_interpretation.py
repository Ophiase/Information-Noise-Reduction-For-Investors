import unittest
import numpy as np
from typing import Dict, Tuple
from information_noise_reduction.interpretation import (
    average_loss_per_variable,
    normalized_contribution_scores,
    softmax_contribution_scores,
    compute_min_max_losses,
    compute_variable_contributions,
    top_k_variables
)
from tests.utils import describe_test

class TestInterpretation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        describe_test("Interpretation")

    def run(self, test):
        result = super().run(test)
        print("\n" + "=" * 30 + "\n")
        return result

    def test_average_loss_per_variable(self):
        results = {('A',): 1.0, ('B',): 2.0, ('A', 'B'): 1.5}
        expected = {'A': 1.25, 'B': 1.75}
        self.assertEqual(average_loss_per_variable(results), expected)

    def test_normalized_contribution_scores(self):
        avg_losses = {'A': 1.25, 'B': 1.75}
        min_loss, max_loss = 1.0, 2.0
        expected = {'A': 0.75, 'B': 0.25}
        self.assertEqual(
            normalized_contribution_scores(avg_losses, min_loss, max_loss), 
            expected)

    def test_compute_min_max_losses(self):
        results = {('A',): 1.0, ('B',): 2.0, ('A', 'B'): 1.5}
        expected_min, expected_max = 1.0, 2.0
        self.assertEqual(compute_min_max_losses(results), (expected_min, expected_max))

    # TODO:
    # def test_softmax_contribution_scores(self):
    #     scores = {'A': 0.25, 'B': 0.0}
    #     expected = {'A': 0.7311, 'B': 0.2689}
    #     result = softmax_contribution_scores(scores)
    #     for key in expected:
    #         self.assertAlmostEqual(result[key], expected[key], places=2)

    # TODO:
    # def test_compute_variable_contributions(self):
    #     results = {('A',): 1.0, ('B',): 2.0, ('A', 'B'): 1.5}
    #     expected = {
    #         'average_losses': {'A': 1.25, 'B': 1.75},
    #         'normalized_scores': {'A': 0.25, 'B': 0.0}
    #     }
    #     result = compute_variable_contributions(results)
    #     self.assertEqual(result['average_losses'], expected['average_losses'])
    #     self.assertEqual(result['normalized_scores'], expected['normalized_scores'])

    def test_top_k_variables(self):
        scores = {'A': 0.25, 'B': 0.75, 'C': 0.5}
        k = 2
        expected = {'B': 0.75, 'C': 0.5}
        self.assertEqual(top_k_variables(scores, k), expected)

if __name__ == "__main__":
    unittest.main()
