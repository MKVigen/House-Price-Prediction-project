import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

#The core of the script focuses on feature engineering and cleaning.
# It uses a Random Forest model to identify the most important features, a technique known as feature selection.
# It then handles missing values by either dropping columns with a high number of missing entries
# or imputing them with a statistical measure like the mean or median. Finally, it uses a StandardScaler to scale the features,

def data_overview(df):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Stats:\n", df.describe())


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=45)
    plt.title("Missing Values per Column")
    plt.show()


def plot_target_distribution(df,target):
    plt.figure(figsize=(10,5))
    sns.histplot(df[target] ,kde=True,bins=30)
    plt.title("Target Distribution")
    plt.show()


def plot_correlation_heatmap(df,target=None):
    num_cols = df.select_dtypes(include="number").columns

    corr = df[num_cols].corr()
    if target:
        corr = corr[[target]].sort_values(by=target,ascending=False)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm',fmt = '.2f')
    plt.title("Correlation Heatmap")
    plt.show()




def feature_importance(df,top_n = 15):
    X = df.drop(columns='SalePrice',axis=1).copy()
    y = df['SalePrice']

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col]).astype(str)

    rf = RandomForestRegressor(n_estimators=100, max_depth=5,random_state=42)
    rf.fit(X,y)

    important = pd.Series(rf.feature_importances_, index=X.columns)
    important = important.sort_values(ascending=False)

    # plt.figure(figsize=(10, 8))
    # sns.barplot(x=important[:top_n], y=important.index[:top_n])
    # plt.title("Top 20 Feature Importances (Random Forest)")
    # plt.show()

    features = important.index[:top_n].tolist()
    return features


def feature_selection(df,features,target = None):
    selected = [f for f in features if f in df.columns]

    if target and target in df.columns:
        selected.append(target)

    df1 = df[selected].copy()
    return df1


def fill_train_missing_values(df):
    if 'GarageFinish' in df.columns:
        df = df.drop(columns=['GarageFinish'],axis=1)
    if 'GarageType' in df.columns:
        df = df.drop(columns=['GarageType'],axis=1)
    if 'Neighborhood' in df.columns:
        df = df.drop(columns=['Neighborhood'],axis=1)
    if 'LotFrontage' in df.columns:
        df = df.fillna(df['LotFrontage'].mean())
    if 'GarageYrBlt' in df.columns:
        df = df.fillna(df['GarageYrBlt'].median())

    return df


def fill_test_missing_values(df):
    if 'GarageFinish' in df.columns:
        df = df.drop(columns=['GarageFinish'],axis=1)
    if 'GarageType' in df.columns:
        df = df.drop(columns=['GarageType'],axis=1)
    if 'Neighborhood' in df.columns:
        df = df.drop(columns=['Neighborhood'],axis=1)
    if 'GarageArea' in df.columns:
        df = df.fillna(df['GarageArea'].median())
    if 'BsmtFinSF1' in df.columns:
        df = df.fillna(df['BsmtFinSF1'].median())
    if 'GarageCars' in df.columns:
        df = df.fillna(df['GarageCars'].median())
    if 'TotalBsmtSF' in df.columns:
        df = df.fillna(df['TotalBsmtSF'].median())

    return df


def train_feature_scaling(trn_df,tst_df,target = 'SalePrice'):
    if target in trn_df.columns:
        x_trn = trn_df.drop(target, axis=1)
        y_trn = trn_df[target]
    else:
        x_trn = trn_df
        y_trn = None

    x_tst = tst_df.copy()

    sc = StandardScaler()
    x_train_scaled = pd.DataFrame(sc.fit_transform(x_trn) , columns=x_trn.columns , index=x_trn.index)
    x_test_scaled = pd.DataFrame(sc.transform(x_tst) , columns=x_tst.columns , index=x_tst.index)

    if y_trn is not None:
        x_train_scaled[target] = y_trn

    return x_train_scaled, x_test_scaled


def save_data(x_train,x_test):
    x_train.to_csv('data/processed/X_train.csv',index=False)
    x_test.to_csv('data/processed/X_test.csv',index=False)



def run_preprocessing():
    # 1. Load data
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')

    # 2. Feature Selection
    top_features = feature_importance(train_df)
    train_df = feature_selection(train_df, top_features, 'SalePrice')
    test_df = feature_selection(test_df, top_features)

    # 3. Handle Missing Values
    train_df = fill_train_missing_values(train_df)
    test_df = fill_test_missing_values(test_df)

    # 4. Scaling
    X_train, X_test = train_feature_scaling(train_df, test_df)

    # 5. Save Processed Data
    save_data(X_train, X_test)
    print("Data preprocessing complete and files saved to 'data/processed/'.")

if __name__ == '__main__':
    run_preprocessing()



