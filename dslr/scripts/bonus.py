import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys


class LogisticRegressionSGD:
    def __init__(self, sgd:str = None, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sgd = sgd

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()

    def _stochastic_gradient_descent(self, X, y):
        m, n = X.shape
        # Add bias term
        X = np.hstack([np.ones((m, 1)), X])
        self.weights = np.random.randn(n + 1)

        for epoch in range(self.epochs):
            loss = 0.0
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y.iloc[random_index:random_index+1].values
                prediction = self.sigmoid(np.dot(xi, self.weights))
                loss += self.loss(int(yi), prediction)
                gradient = xi.T.dot(prediction - yi)
                # print(gradient.shape)
                self.weights -= self.learning_rate * gradient
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss/m}")

    def _batch_gradient_descent(self, X, y):
        m, n = X.shape
        # Add bias term
        X = np.hstack([np.ones((m, 1)), X])
        self.weights = np.random.randn(n + 1)

        for epoch in range(self.epochs):
            prediction = self.sigmoid(np.dot(X, self.weights))
            loss = self.loss(y, prediction).mean()
            gradient = X.T.dot(prediction - y)
            self.weights -= self.learning_rate * gradient/m
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")

    def train(self, X, y):
        if self.sgd == 'stochastic':
            self._stochastic_gradient_descent(X, y)
        elif self.sgd == 'batch':
            self._batch_gradient_descent(X, y)
        else:
            print("Please specify a valid SGD method.")
            exit(1)

    def predict(self, X):
        # Add bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.sigmoid(np.dot(X, self.weights)) >= 0.5


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

    # Train the model using SGD
    models_sgd = {}
    houses = y_train.unique()
    for house in houses:
        model = LogisticRegressionSGD(epochs=300, sgd='batch', learning_rate=0.5)
        y_binary = y_train == house
        model.train(X_train_preprocessed, y_binary)
        models_sgd[house] = model

    # Save the trained weights to a file
    weights = {house: model.weights for house, model in models_sgd.items()}
    np.save('trained_weights.npy', weights)

print("Training completed and weights saved.")
