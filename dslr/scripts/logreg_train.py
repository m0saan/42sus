import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys

from sklearn.linear_model import LogisticRegression
# from sklearn.multiclass import OneVsAllClassifier


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

def train(X, y):
    ova_classifier = OneVsOneClassifier(LogisticRegression())
    ova_classifier.fit(X_train_preprocessed, y_train)
    return ova_classifier




if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <train_data.csv>")
        exit(1)

    # Load training data
    train_data = pd.read_csv(sys.argv[1])

    # Preprocess the data
    X_train = train_data.drop(
        columns=["Hogwarts House", "First Name", "Last Name", "Birthday", "Index"])
    y_train = train_data["Hogwarts House"]

    # Define numerical and categorical columns
    numerical_cols = X_train.select_dtypes(
        include=["float64", "int64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object"]).columns.tolist()

    # Preprocessing for numerical data: imputation and scaling
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: imputation and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Preprocess the training data and save the preprocessor
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    pkl.dump(preprocessor, open('preprocessor.pkl', 'wb'))

    model = train(X_train_preprocessed, y_train)
    

    # Save the trained weights to a file
    pkl.dump(model, open('model.pkl', 'wb'))

print("Training completed and weights saved.")