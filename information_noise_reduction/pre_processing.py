from typing import List
import pandas as pd
from sklearn.linear_model import  Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression


def select_important_features_with_lasso(df: pd.DataFrame, target_col: str, k: int = 20) -> List[str]:
    """
    Applies Lasso regression with cross-validation to select important features, 
    then retains the k most significant features after redundancy testing.
    
    :param X: DataFrame of features
    :param y: Target variable
    :param k: Number of top features to select after redundancy testing
    :return: List of top k important features selected by the Lasso model
    """
    X, y = df.drop(columns=[target_col]), df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = Lasso()
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10]}
    lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error')
    lasso_grid.fit(X_scaled, y)
    
    best_lasso = lasso_grid.best_estimator_
    lasso_coefficients = pd.Series(best_lasso.coef_, index=X.columns)
    important_features = lasso_coefficients[lasso_coefficients != 0].index.tolist()
    
    X_important = X[important_features]
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_important, y)
    top_k_features = X_important.columns[selector.get_support()].tolist()
    
    return top_k_features

