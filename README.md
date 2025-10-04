## 🏡 House Price Prediction Project (Kaggle Competition)
### 🌟 Overview

This project aims to accurately predict house sale prices using various machine learning models. It follows a standard data science pipeline: Feature Engineering, Data Preprocessing, Model Training & Evaluation, and Hyperparameter Tuning.

The primary goal is to compare the performance of several regression models, including advanced ensemble and boosting techniques, to find the optimal predictive solution.

### 🚀 Getting Started
Follow these steps to set up the environment and reproduce the analysis.

#### Prerequisites 
You need Python 3.8+ installed. The project relies on the following libraries:

##### Bash
pip install pandas scikit-learn matplotlib seaborn xgboost lightgbm
For full dependency management, use the provided requirements.txt file (you should create this):

##### Bash
pip install -r requirements.txt
### 📁 Project Structure
This repository uses a structured layout:

```House Prices project/
├── .vscode/                 # IDE-specific configuration (typically ignored)
├── House_Prices_env/        # Python virtual environment (MUST be ignored by Git)
├── data/                    # Contains all data assets.
│   └── processed/           # Current location for all data files:
│       ├── sample_submission.csv
│       ├── test.csv         # Raw Test Data
│       └── train.csv        # Raw Training Data
├── notebooks/               # Jupyter notebooks for exploratory analysis and prototyping.
│   ├── EDX.ipynb
│   └── modeling.ipynb
├── src/                     # All executable Python source code lives here.
│   ├── data_preprocessing.py    # Script for Feature Selection, Imputation, and Scaling.
│   ├── model_train.py           # Script for training and evaluating multiple models.
│   ├── model_tuning.py          # Script for Hyperparameter Tuning.
│   └── main.py                  # Project entry point.
├── submissions/             # Output directory for final model predictions.
├── .gitignore               # Essential file listing ignored paths.
├── README.md                # This document.
└── requirements.txt         # List of Python dependencies.
```

#### ⚙️ Usage and Running the Scripts
The pipeline is designed to be run sequentially:

#### Step 1: Data Preprocessing
This script loads the raw data, performs feature selection (using Random Forest feature importance), imputes missing values, and scales the features.

##### Bash
python3 data_preprocessing.py
Output: Two files (X_train.csv and X_test.csv) will be saved to the data/processed/ directory.

#### Step 2: Model Training and Comparison
This script loads the processed data, trains seven different models (Random Forest, XGBoost, LGBM, Simple Linear, Lasso, Ridge, and a Stacking Regressor), evaluates them using the R^2 score on a validation set, generates predictions, and saves the submission files.

##### Bash
python3 model_train.py

Output:Prints the R^2 score for each model to the console.
Saves submission files (e.g., submissionXGBoost.csv) to the submissions/ directory.
(Optional: If you uncomment the plotting code) Displays a bar chart comparing model R^2 scores.

#### Step 3: Hyperparameter Tuning (Optional)
This script performs a Randomized Search Cross-Validation to find optimal hyperparameters for the XGBoost and LGBM models, which typically achieve the best performance.

##### Bash
python3 model_tuning.py

Output: Prints the best parameters and their corresponding R^2 scores achieved during the search.


### 📈 Key Results
#### Model	Validation Set R^2

Score (Example)	Description
Model	Validation Set R^2

### Score Description

LGBM Regressor	0.9063 ---Highest performing individual model; fast training and strong predictive power.

XGBoost	0.8999 --- Excellent performance from the core gradient boosting machine.

Stacking Regressor	0.8995 ---A robust combination of ensemble models, confirming strong predictive stability.

Random Forest 0.8938 --- Strong performance from the bagging ensemble baseline model.

Ridge Regression	0.8723 ---Linear model using L2 regularization for stability.

Lasso Regression	0.8705 ---Linear model using L1 regularization for feature selection and stability.

Linear Regression	0.8703 ---The basic linear model, providing a solid baseline score.

#### 🧑‍💻 Author
Name Vigen Mkrtchyan  

GitHub [https://github.com/MKVigen]
#### 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.