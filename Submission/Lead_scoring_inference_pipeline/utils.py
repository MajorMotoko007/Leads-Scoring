'''
filename: utils.py
functions: encode_features, load_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd

import sqlite3

import os
import logging
from constants import *
from datetime import datetime
from pycaret.classification import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

###############################################################################
# Define the function to train the model
# ##############################################################################


def encode_features():
    # Load the data from the specified database file
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    print(db_file_path)
    conn = sqlite3.connect(db_file_path)
    data = pd.read_sql("SELECT * FROM model_input", conn)
    conn.close()

    encoded_df = pd.DataFrame(columns= ONE_HOT_ENCODED_FEATURES)
    placeholder_df = pd.DataFrame()
# One-Hot Encoding using get_dummies for the specified categorical features
    for f in FEATURES_TO_ENCODE:
        if(f in data.columns):
            print(f)
            encoded = pd.get_dummies(data[f])
            encoded = encoded.add_prefix(f +'_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')

    # Implement these steps to prevent dimension mismatch during inference
    for feature in encoded_df.columns:
        if feature in data.columns:
            encoded_df[feature] = df[feature]
        if feature in placeholder_df.columns:
            encoded_df[feature] = placeholder_df[feature]
    # fill all null values
    encoded_df.fillna(0, inplace=True)
    # Combine the encoded features with the rest of the data
    print(encoded_df.info())
    # Save the encoded features in a table named 'features'
    conn = sqlite3.connect(db_file_path)
    encoded_df.to_sql('features', conn, index=False, if_exists='replace')
    conn.close()

###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################

def get_models_prediction():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load the model from MLflow model registry
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{STAGE}")

    # Load the input dataset
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    input_data = pd.read_sql("SELECT * FROM features", conn)  # Replace 'your_input_table' with the actual table name
    conn.close()
    
    with open("/home/Assignment/03_inference_pipeline/scripts/log_file.txt", 'a') as file:
        file.write(",".join(model.feature_name_))
    
    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    # Store the predicted values along with input data into a table named 'predicted_values'
    output_data = pd.concat([input_data, pd.DataFrame({'predictions': predictions})], axis=1)
    conn = sqlite3.connect(db_file_path)
    output_data.to_sql('predicted_values', conn, index=False, if_exists='replace')
    conn.close()

###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################

def prediction_ratio_check():
    # Load the predicted values from the 'predicted_values' table
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    predicted_data = pd.read_sql("SELECT * FROM predicted_values", conn)
    conn.close()

    # Calculate the percentage of 1s and 0s
    percentage_1 = (predicted_data['predictions'] == 1).mean() * 100
    percentage_0 = (predicted_data['predictions'] == 0).mean() * 100

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the results to 'prediction_distribution.txt'
    with open("/home/Assignment/airflow/dags/Lead_scoring_inference_pipeline/prediction_distribution.txt", 'a') as file:
        file.write(f"{timestamp} - Percentage of 1s: {percentage_1:.2f}% | Percentage of 0s: {percentage_0:.2f}%\n")

###############################################################################
# Define the function to check the columns of input features
# ##############################################################################
   

def input_features_check():
    # Load the input features from the 'features' table
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    input_data = pd.read_sql("SELECT * FROM features", conn)
    conn.close()

    # Get the list of columns present in the input data
    input_columns = input_data.columns.tolist()

    # Check if all ONE_HOT_ENCODED_FEATURES are present in the input data
    missing_columns = set(ONE_HOT_ENCODED_FEATURES) - set(input_columns)

    # Create a log message based on the presence of all columns
    if not missing_columns:
        log_message = "All the model inputs are present"
    else:
        log_message = f"Some of the model inputs are missing: {', '.join(missing_columns)}"

    # Write the log message to a log file
    with open("/home/Assignment/log_file.txt", 'a') as file:
        file.write(log_message + "\n")
   