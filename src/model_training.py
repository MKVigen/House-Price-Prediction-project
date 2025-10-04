import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error ,r2_score ,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import StackingRegressor


#Each model is trained and evaluated using the R^2 score on a validation set.
#The script then generates predictions for the test data and saves them to individual CSV files.
# Finally, it uses a bar chart to visually compare the performance (R^2scores) of all the models,
# providing a clear summary of which performed best.

def read_data():
    x_train = pd.read_csv('data/processed/X_train.csv')
    x_test = pd.read_csv('data/processed/X_test.csv')
    data = pd.read_csv('data/raw/test.csv')

    return x_train,x_test , data

X_train,X_test ,data = read_data()


def RandomForest(x_train,x_test,data):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']
    x_train ,x_val , y_train ,y_val = train_test_split(x,y,test_size=0.2 ,random_state=42)


    model = RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    loss = root_mean_squared_error(y_pred,y_val)
    r2 = r2_score(y_val,y_pred)
    # print(f'loss: {loss}')
    # print(f'r2_score: {r2}')

    prediction = model.predict(x_test)

    submission = pd.DataFrame({
        "Id": data["Id"],
        "SalePrice": prediction
    })

    submission.to_csv("submissions/submissionRandomForest.csv", index=False)

    return r2

# r2 = RandomForest(X_train,X_test,data)


def XGBoost(x_train,x_test,data):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    x_train , x_val , y_train ,y_val = train_test_split(x,y,test_size=0.2 ,random_state=42)
    model = XGBRegressor(n_estimators=200,learning_rate=0.05 ,max_depth=3, subsample=0.7,colsample_bytree=0.8,random_state=42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    loss = root_mean_squared_error(y_pred,y_val)
    r2 = r2_score(y_val,y_pred)
    # print(f'loss: {loss}')
    # print(f'r2_score: {r2}')

    prediction = model.predict(x_test)
    submission = pd.DataFrame({
        'Id' : data['Id'],
        'SalePrice': prediction
    })

    submission.to_csv("submissions/submissionXGBoost.csv", index=False)
    return r2

# r2 = XGBoost(X_train,X_test,data)

def train_LGBMRegressor(x_train,x_test,data):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    x_train,x_val , y_train ,y_val = train_test_split(x,y,test_size=0.2 ,random_state=42)
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=6,
        random_state=42,
        verbose=-1,
    )

    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    loss = root_mean_squared_error(y_pred,y_val)
    r2 = r2_score(y_val,y_pred)
    # print(f'loss: {loss}')
    # print(f'r2_score: {r2}')

    prediction = model.predict(x_test)
    submission = pd.DataFrame({
        'Id' : data['Id'],
        'SalePrice': prediction
    })
    submission.to_csv("submissions/submissionLGBM.csv", index=False)
    return r2

r2 = train_LGBMRegressor(X_train,X_test,data)


def simple_linear_regressor(x_train,x_test,data):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    x_train,x_val , y_train ,y_val = train_test_split(x,y,test_size=0.2 ,random_state=42)

    poly = PolynomialFeatures(degree=2,include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_val_poly = poly.transform(x_val)
    x_test_poly = poly.transform(x_test)


    model = LinearRegression()
    model.fit(x_train_poly,y_train)
    y_pred = model.predict(x_val_poly)
    loss = root_mean_squared_error(y_pred,y_val)
    r2 = r2_score(y_val,y_pred)
    # print(f'loss: {loss}')
    # print(f'r2_score: {r2}')

    prediction = model.predict(x_test_poly)
    submission = pd.DataFrame({
        'Id' : data['Id'],
        'SalePrice': prediction
    })

    submission.to_csv("submissions/submissionSimpleLinear.csv", index=False)
    return r2

# r2 = simple_linear_regressor(X_train,X_test,data)


#this model as showed r2 doesn't give apropriate result because features dont depand on each other linearly and they need
#polynomial transformation
def lasso_regressor(x_train,x_test,data):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    x_train , x_val , y_train ,y_val = train_test_split(x,y,test_size=0.2 ,random_state=42)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_val_poly = poly.transform(x_val)
    x_test_poly = poly.transform(x_test)

    model = Lasso(alpha=0.01 , max_iter=5000)
    model.fit(x_train_poly,y_train)
    y_pred = model.predict(x_val_poly)
    loss = root_mean_squared_error(y_pred,y_val)
    r2 = r2_score(y_val,y_pred)
    # print(f'loss: {loss}')
    # print(f'r2_score: {r2}')

    prediction = model.predict(x_test_poly)
    submission = pd.DataFrame({
        'Id' : data['Id'],
        'SalePrice': prediction
    })
    submission.to_csv("submissions/submissionLasso.csv", index=False)
    return r2

# r2 = lasso_regressor(X_train,X_test,data)


def ridge_regressor(x_train,x_test,data):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    x_train , x_val , y_train ,y_val = train_test_split(x,y,test_size=0.2 ,random_state=42)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_val_poly = poly.transform(x_val)
    x_test_poly = poly.transform(x_test)

    model = Ridge(alpha=0.1 , max_iter=1000)
    model.fit(x_train_poly,y_train)
    y_pred = model.predict(x_val_poly)
    loss = root_mean_squared_error(y_pred,y_val)
    r2 = r2_score(y_val,y_pred)
    # print(f'loss: {loss}')
    # print(f'r2_score: {r2}')

    prediction = model.predict(x_test_poly)
    submission = pd.DataFrame({
        'Id' : data['Id'],
        'SalePrice': prediction
    })

    submission.to_csv("submissions/submissionRidge.csv", index=False)
    return r2

# r2 = ridge_regressor(X_train,X_test,data)


def stacking_model(x_train,x_test,data):
    x = x_train.drop('SalePrice', axis=1)
    y = x_train['SalePrice']

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state = 42)

    xgb_model = XGBRegressor(n_estimators=200,learning_rate=0.05 ,max_depth=3, subsample=0.7,colsample_bytree=0.8,random_state=42)
    lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42,verbose=-1)
    rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)

    stack = StackingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('lgbm', lgbm_model),
            ('rf', rf_model)
        ],
        final_estimator=Ridge(alpha=0.1,max_iter=10000),
        n_jobs=-1
    )

    stack.fit(x_train, y_train)
    y_pred = stack.predict(x_val)
    r2 = r2_score(y_val, y_pred)
    loss = np.sqrt(mean_squared_error(y_val, y_pred))

    # print("R2 score:",r2)
    # print("RMSE:", loss)

    prediction = stack.predict(x_test)
    submission = pd.DataFrame({
        'Id' : data['Id'],
        'SalePrice': prediction
    })

    submission.to_csv("submissions/submissionStacking.csv", index=False)

    return r2


# stacking_model(X_train,X_test,data)

def model_r2_score(X_train,X_test,data):
    r2_rf = RandomForest(X_train,X_test,data)
    r2_xgb = XGBoost(X_train,X_test,data)
    r2_lgbm = train_LGBMRegressor(X_train,X_test,data)
    r2_linreg = simple_linear_regressor(X_train,X_test,data)
    r2_lasso = lasso_regressor(X_train,X_test,data)
    r2_ridge = ridge_regressor(X_train,X_test,data)
    r2_stack = stacking_model(X_train,X_test,data)

    scores = {}
    scores['random forest regression'] = r2_rf
    scores['XGBoost'] = r2_xgb
    scores['LGBMRegressor']=r2_lgbm
    scores['linear regression']=r2_linreg
    scores['lasso regression'] = r2_lasso
    scores['ridge regression'] = r2_ridge
    scores['stacking']=r2_stack

    for model , score in scores.items():
        print(f'{model}: {score}' , end='\n')

    return scores



def model_comparing(scores):
    plt.figure(figsize=(8, 6))
    plt.bar(scores.keys(), scores.values(), color="skyblue")
    plt.ylabel("R² Score")
    plt.title("Model Comparison (R² on Validation Set)")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.show()


def run_model_training():
    X_train, X_test, data = read_data()
    scores = model_r2_score(X_train, X_test, data)
    # model_comparing(scores)

if __name__ == '__main__':
    run_model_training()





















