# Titanic-Survival-Prediction-ML-model

🚢 Titanic Survival Prediction using Machine Learning Pipeline
📌 Project Overview

The Titanic Survival Prediction project aims to predict whether a passenger survived the Titanic disaster using supervised machine learning.
This project is implemented using Scikit-Learn Pipelines, ensuring a clean, modular, scalable, and production-ready workflow.

The pipeline automates:

Data preprocessing

Feature engineering

Feature selection

Model training

Hyperparameter tuning

Model evaluation
🎯 Objective

To build a robust ML model that predicts passenger survival based on demographic and travel-related features while following best ML engineering practices.

📂 Dataset Information

Dataset: Titanic Dataset (Kaggle)

Target Variable: Survived

0 → Did not survive

1 → Survived

❌ Dropped Columns
PassengerId, Name, Ticket, Cabin


These columns either contain excessive missing values or do not contribute meaningfully to prediction.

🧰 Libraries & Tools Used
🔹 Core Libraries
numpy
pandas
scikit-learn

🔹 Scikit-Learn Modules

Data Splitting: train_test_split

Pipelines: Pipeline, make_pipeline

Preprocessing:

ColumnTransformer

SimpleImputer

OneHotEncoder

MinMaxScaler

Feature Selection: SelectKBest, chi2

Model: DecisionTreeClassifier

Evaluation:

accuracy_score

cross_val_score

Hyperparameter Tuning: GridSearchCV

🔁 End-to-End Machine Learning Pipeline
🧠 Pipeline Process Diagram
                ┌────────────────────┐
                │   Raw Titanic Data  │
                └─────────┬──────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │   Train / Test Split (80/20)│
            └─────────┬──────────────────┘
                      │
        ┌─────────────┴─────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐        ┌──────────────────────┐
│ Numerical Features│        │ Categorical Features │
│ (Age, Fare, etc.) │        │ (Sex, Embarked, etc.)│
└─────────┬────────┘        └──────────┬───────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐        ┌────────────────────────┐
│ Mean Imputation  │        │ Most-Frequent Imputer  │
└─────────┬────────┘        └──────────┬─────────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐        ┌────────────────────────┐
│ MinMax Scaler    │        │ One-Hot Encoding       │
└─────────┬────────┘        └──────────┬─────────────┘
          └──────────────┬─────────────┘
                         ▼
              ┌─────────────────────────┐
              │  ColumnTransformer       │
              └─────────┬───────────────┘
                        ▼
              ┌─────────────────────────┐
              │ SelectKBest (Chi-Square) │
              └─────────┬───────────────┘
                        ▼
              ┌─────────────────────────┐
              │ Decision Tree Classifier │
              └─────────┬───────────────┘
                        ▼
              ┌─────────────────────────┐
              │ Model Prediction Output  │
              └─────────────────────────┘

🛠 Pipeline Stages Explained
1️⃣ Train-Test Split

80% Training

20% Testing

Ensures unbiased model evaluation.

2️⃣ Data Preprocessing
🔹 Numerical Features

Missing values → Mean Imputation

Scaling → MinMaxScaler (0–1 range)

🔹 Categorical Features

Missing values → Most Frequent

Encoding → OneHotEncoder

3️⃣ ColumnTransformer

Combines numerical and categorical preprocessing into a single unified step, ensuring clean data flow.

4️⃣ Feature Selection
SelectKBest(score_func=chi2)


Selects the most relevant features

Reduces noise and overfitting

5️⃣ Model Training
DecisionTreeClassifier


Interpretable model

Handles non-linear patterns efficiently

📊 Model Evaluation
✅ Accuracy Score

Evaluated on test data using:

accuracy_score(y_test, y_pred)

🔁 Cross Validation
cross_val_score(pipe, X_train, y_train, cv=5)


5-fold cross validation

Improves generalization reliability

🔍 Hyperparameter Tuning
GridSearchCV Parameters
selectkbest__k = [5, 8, 10]
decisiontreeclassifier__max_depth = [3, 5, None]

Benefits

Automatically finds best model

Prevents underfitting & overfitting

Improves accuracy consistency

🏆 Final Results

Test Accuracy: ~ Optimized via GridSearchCV

Cross-Validated Accuracy: Mean of 5 folds

Pipeline ensures no data leakage

▶️ How to Run the Project
pip install numpy pandas scikit-learn
jupyter notebook titanic-using-pipeline.ipynb
