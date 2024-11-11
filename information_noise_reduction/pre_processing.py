import pandas as pd
from sklearn.linear_model import  Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



def select_important_features_with_lasso(X,y):
    """
    Applies Lasso regression with cross-validation to select important features for predicting bankruptcy.
    :param df: X (features), y (target variable)
    :return: List of important features selected by the Lasso model
    """
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = Lasso()
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10]}
    lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error')
    lasso_grid.fit(X_scaled, y)
    
    best_lasso = lasso_grid.best_estimator_
    lasso_coefficients = pd.Series(best_lasso.coef_, index=X.columns)
    important_features = lasso_coefficients[lasso_coefficients != 0].index.tolist()
    
    return important_features
