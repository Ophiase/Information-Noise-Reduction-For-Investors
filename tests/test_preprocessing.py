import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from information_noise_reduction.pre_processing import select_important_features_with_lasso
from tests.utils import describe_test

class TestSubsetGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        describe_test("PreProcessing")

    def run(self, test):
        result = super().run(test)
        print("\n" + "=" * 30 + "\n")
        return result

    def test_select_important_features_with_lasso(self):
        np.random.seed(42)
        
        X, y, coef = make_regression(
            n_samples=800,
            n_features=10,
            n_informative=5,
            noise=0.1,
            coef=True,
            random_state=42
        )
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        k = 5
        selected_features = select_important_features_with_lasso(df, target_col="target", k=k)
        
        lasso_top_features = [
            feature_names[i] for i in np.argsort(-np.abs(coef))[:k]
        ]
        
        missing_features = set(lasso_top_features) - set(selected_features)
        unexpected_features = set(selected_features) - set(lasso_top_features)

        self.assertTrue(len(missing_features) == 0, f"Missing features: {missing_features}")
        self.assertTrue(len(unexpected_features) == 0, f"Unexpected features: {unexpected_features}")

if __name__ == "__main__":
    unittest.main()
