# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using the training dataset and evaluates using the test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree.')


    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from CSV
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Split the data into input(X) and output(y)
    y_train = train_df['price']
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Initialize and train a Decision Tree Classifier
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict target values on the test dataset using the trained model, and calculate the mean squared error. 
    yhat_test = model.predict(X_test)

    # Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.
    mse = mean_squared_error(y_test, yhat_test)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    mlflow.log_metric("MSE", float(mse))  # Log the MSE

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
