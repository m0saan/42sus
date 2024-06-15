
import sys
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <test_data.csv>")
        exit(1)

    # Load test data
    test_data = pd.read_csv(sys.argv[1])

    # Extract true labels and features
    X_test = test_data.drop(
        columns=["Hogwarts House", "First Name", "Last Name", "Birthday", "Index"])
    y_true = test_data["Hogwarts House"]

    # Define numerical and categorical columns
    numerical_cols = X_test.select_dtypes(
        include=["float64", "int64"]).columns.tolist()
    categorical_cols = X_test.select_dtypes(
        include=["object"]).columns.tolist()

    # Load preprocessor
    preprocessor = pkl.load(open('preprocessor.pkl', 'rb'))
    X_test_preprocessed = preprocessor.transform(X_test)

    # Load trained weights
    model = pkl.load(open('model.pkl', 'rb'))
    preds = model.predict(X_test_preprocessed)
    
    # Save the predictions to houses.csv
    houses_output = pd.DataFrame({
        "Index": test_data["Index"],
        "Hogwarts House": preds
    })
    houses_output.to_csv('houses.csv', index=False)
    
    
    # calculate accuracy
    truth_df = pd.read_csv('data/dataset_truth.csv')
    acc = (preds == truth_df['Hogwarts House']).mean()
    print("Accuracy: {}".format(acc))   

print("Predictions saved to houses.csv.")
