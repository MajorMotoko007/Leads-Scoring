'''
filename: utils.py
functions: encode_features, get_train_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sqlite3
from sqlite3 import Error
from pycaret.classification import *
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV # run pip install scikit-optimize
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from datetime import datetime
from datetime import date

from constants import *


###############################################################################
# Define the function to encode features
# ##############################################################################

def build_dbs():
    if os.path.isfile(ML_DB_PATH+MLFLOW_DB_NAME):
        print( "DB Already Exsist")
        print(os.getcwd())
        return "DB Exsist"
    else:
        print ("Creating Database")
        """ create a database connection to a SQLite database """
        conn = None
        try:
            
            conn = sqlite3.connect(ML_DB_PATH+MLFLOW_DB_NAME)
            print("New DB Created")
        except Error as e:
            print(e)
            return "Error"
        finally:
            if conn:
                conn.close()
                return "DB Created"


def encode_features():
    db_file_path = db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    print(db_file_path)
    conn = sqlite3.connect(db_file_path)
    # Load the dataset
    query = "SELECT * FROM model_input"
    data = pd.read_sql(query, conn)

    # Select only the features to be one-hot encoded
    features_to_encode = data[FEATURES_TO_ENCODE]

    # Perform one-hot encoding
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_features = encoder.fit_transform(features_to_encode)

    # Get the names of the one-hot encoded features manually
    feature_names = []
    for i, feature in enumerate(FEATURES_TO_ENCODE):
        unique_categories = data[feature].unique()
        for category in unique_categories[1:]:  
            feature_names.append(f"{feature}_{category}")

    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names)

    # Save the encoded features to an SQLite database
    db_file_path = db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    print(db_file_path)
    conn = sqlite3.connect(db_file_path)
    encoded_df.to_sql('features', conn, index=False, if_exists='replace')
    #encoded_df.to_csv('encodec.csv',index=False)

    # Save the target variable to the same SQLite database
    target = data['app_complete_flag']
    target_df = pd.DataFrame({'target': target})
    target_df.to_sql('target', conn, index=False, if_exists='replace')
   # target_df.to_csv('target.csv',index=False)
    # Close the connection
    conn.close()

###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model():
    # Set MLflow tracking URI and create the experiment if it doesn't exist
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Load the entire 'model_input' table from SQLite
    db_file_path = os.path.join(ML_DB_PATH, MLFLOW_DB_NAME)
    # Load the entire 'model_input' table from SQLite
    db_url = f"sqlite:///{DB_PATH}/{DB_FILE_NAME}"
    
    # Load features and target from the 'features' and 'target' tables in the database
    features_df = pd.read_sql_table('features', db_url)
    target_df = pd.read_sql_table('target', db_url)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=42)

    # Initialize LightGBM model (customize parameters as needed)
    model = lgb.LGBMClassifier()

    # Train the model
    model.fit(X_train, y_train)
    
    

    # Log parameters
    mlflow.log_param(db_file_path, DB_PATH)

    # Log the model into MLflow model registry
    mlflow.lightgbm.log_model(model, "LightGBM")
    # Register the model in MLflow
    mlflow.register_model(model_uri=f"runs:/{mlflow.active_run().info.run_id}/LightGBM", name="LightGBM")

    # Make predictions on the test set
    predictions = model.predict_proba(X_test)[:, 1]

    # Calculate AUC and log it as a metric
    auc_score = roc_auc_score(y_test, predictions)
    mlflow.log_metric("AUC", auc_score)

    # End the MLflow run
    mlflow.end_run()
    

def get_validation_unseen_set(dataframe, validation_frac=0.05, sample=False, sample_frac=0.1):
    if not sample:
        dataset = dataframe.copy()
    else:
        dataset = dataframe.sample(frac=sample_frac)
    data = dataset.sample(frac=(1-validation_frac), random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(inplace=True, drop=True)
    data_unseen.reset_index(inplace=True, drop=True)
    return data, data_unseen

def get_transformation_pipeline_from_setup():
    return get_config(variable="prep_pipe")

   
