from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pathlib import Path

"""
Hyperparameter Tuning Module

This module is dedicated to finding the optimal hyperparameter settings for
the high-performing models (XGBoost and LightGBM) using cross-validation.

Tuning Strategy:
1.  **RandomizedSearchCV**: This approach is favored for its computational efficiency, 
    as it explores a broad range of the hyperparameter space by sampling combinations 
    randomly. This method is generally faster and often sufficient for finding near-optimal settings.
2.  **GridSearchCV**: This method exhaustively searches all possible combinations in the 
    defined grid. It is guaranteed to find the absolute best combination within the grid 
    but is computationally very expensive and time-consuming, especially with large search spaces.

We prioritize RandomizedSearchCV to balance performance maximization with resource usage.
"""

def read_data():
    base_dir = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd().parent

    x_train_path = base_dir / 'data' /'processed'
    x_test_path = base_dir / 'data' /'processed'
    data_path = base_dir / 'data'/'raw'

    x_train = pd.read_csv(x_train_path /'X_train.csv')
    x_test = pd.read_csv(x_test_path/'X_test.csv')
    data = pd.read_csv(data_path/'test.csv')

    return x_train,x_test , data

X_train,X_test ,data = read_data()

def randomized_search_CV(x_train):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    xgb = XGBRegressor(random_state=42)

    param_grid={
        'n_estimators': [100,300,500],
        'learning_rate' : [0.01 , 0.05 ,0.1],
        'max_depth' : [3,5,7],
        'subsample' : [0.7,0.8,1],
        'colsample_bytree' : [0.7,0.8,1]
    }

    xgb_search = RandomizedSearchCV(
        xgb , param_distributions=param_grid,
        n_iter=20,cv=5,scoring='r2',verbose=1,n_jobs=-1,random_state=42
    )

    xgb_search.fit(x,y)
    print("Best parameters:", xgb_search.best_params_)
    print("Best R2 score:", xgb_search.best_score_)

    return xgb_search.best_params_


best_params =  randomized_search_CV(X_train)

def randomized_search_CV(x_train):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    lgbm = LGBMRegressor(random_state=42)

    param_grid={
        'n_estimators': [100,300,500],
        'learning_rate' : [0.01 , 0.06 ,0.1],
        'max_depth' : [4,6,8],
        'subsample' : [0.7,0.8,1],
        'colsample_bytree' : [0.7,0.8,1],
        'min_child_samples' : [3,5,7],
        'min_split_gain' : [0.01,0.05,0.1]
    }

    lgbm_search = RandomizedSearchCV(
        lgbm , param_distributions=param_grid,
        n_iter=20,cv=5,scoring='r2',verbose=0,n_jobs=-1,random_state=42,
    )

    lgbm_search.fit(x,y)
    print("Best parameters:", lgbm_search.best_params_)
    print("Best R2 score:", lgbm_search.best_score_)

    return lgbm_search.best_params_

best_params_lgbm = randomized_search_CV(X_train)


def grid_search_CV_xgb(x_train):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    xgb = XGBRegressor(random_state=42)
    param_grid={
        'n_estimators' : [100,300,500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1],
        'colsample_bytree': [0.7, 0.8, 1]
    }

    grid_search = GridSearchCV(xgb, param_grid,n_jobs=-1)
    grid_search.fit(x,y)

    print("Best parameters:", grid_search.best_params_)
    print("Best R2 score:", grid_search.best_score_)

# grid_search_CV_xgb(X_train)


