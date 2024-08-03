import numpy as np
import pandas as pd
import os
import sys
from constants import *
from schema import raw_data_schema,model_input_schema
import sqlite3
from sqlite3 import Error
from mapping.significant_categorical_level import list_platform, list_medium, list_source
from mapping.city_tier_mapping import city_tier_mapping

###############################################################################
# Define function to validate raw data's schema
############################################################################### 

def raw_data_schema_check():
    # Construct the full path of the CSV file
    csv_file_path = os.path.join(DATA_DIRECTORY, 'leadscoring.csv')

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Get the columns from the DataFrame
    data_columns = list(df.columns)

    # Check if all columns from raw_data_schema are present in data_columns
    missing_columns = [col for col in raw_data_schema if col not in data_columns]

    # Print the result
    if not missing_columns:
        print("Raw data schema is in line with the schema present in schema.py")
    else:
        print("Raw data schema is NOT in line with the schema present in schema.py")
        print("Missing columns:", missing_columns)


###############################################################################
# Define function to validate model's input schema
############################################################################### 

def model_input_schema_check():
    db_file_path = f'{DB_PATH}/{DB_FILE_NAME}'

    # Connect to the database
    conn = sqlite3.connect(db_file_path)

    # Get the columns from the 'model_input' table
    query = 'PRAGMA table_info(model_input)'
    cursor = conn.execute(query)
    model_input_columns = [column[1] for column in cursor.fetchall()]
    print(model_input_columns)
    
    print(model_input_schema)
    # Check if all columns in the model_input_schema are present in the 'model_input' table
    if set(model_input_schema).issubset(set(model_input_columns)):
        print("Models input schema is in line with the schema present in schema.py")
    else:
        print("Models input schema is NOT in line with the schema present in schema.py")

    # Close the connection
    conn.close()
    

    
    
