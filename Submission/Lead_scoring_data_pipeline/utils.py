##############################################################################
# Import necessary modules and files
##############################################################################
import pandas as pd
import os
import sqlite3
from sqlite3 import Error
from mapping.significant_categorical_level import list_platform, list_medium, list_source
from constants import *
from mapping.city_tier_mapping import city_tier_mapping
###############################################################################
# Define the function to build database
###############################################################################

def calculate_significant_levels(df, column):
    """
    Calculate significant levels in the specified column based on cumulative percentage.

    Parameters:
    - df: DataFrame containing the data.
    - column: Name of the column for which significant levels are calculated.

    Returns:
    - significant_levels: List of significant levels.
    """
    df_cat_freq = df[column].value_counts()
    df_cat_freq = pd.DataFrame({'column': df_cat_freq.index, 'value': df_cat_freq.values})
    df_cat_freq['perc'] = df_cat_freq['value'].cumsum() / df_cat_freq['value'].sum()

    significant_levels = list(df_cat_freq.loc[df_cat_freq['perc'] <= 0.9].column)

    return significant_levels


def build_dbs():
    if os.path.isfile(DB_PATH+DB_FILE_NAME):
        print( "DB Already Exsist")
        print(os.getcwd())
        return "DB Exsist"
    else:
        print ("Creating Database")
        """ create a database connection to a SQLite database """
        conn = None
        try:
            
            conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
            print("New DB Created")
        except Error as e:
            print(e)
            return "Error"
        finally:
            if conn:
                conn.close()
                return "DB Created"
###############################################################################
# Define function to load the csv file to the database
###############################################################################

def load_data_into_db():
    # Construct the full path of the database file
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)

    # Read the CSV file into a DataFrame
    csv_file_path = os.path.join(DATA_DIRECTORY, 'leadscoring.csv')
    df = pd.read_csv(csv_file_path,index_col=[0])

    # Replace null values in specific columns with 0
    columns_to_replace_null = ['total_leads_droppped', 'referred_lead']
    df[columns_to_replace_null] = df[columns_to_replace_null].fillna(0)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    
    print('leadscoring '+df.columns.values)
    # Save the processed DataFrame to the database
    df.to_sql('loaded_data', conn, index=False, if_exists='replace')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    


def replace_for_score_with_others(df, column):
    """
    Replace levels in the specified column that are not present in list_levels
    with the specified replacement_value.

    Parameters:
    - df: DataFrame to perform the replacement on.
    - column: Name of the column where levels are to be replaced.
    - list_levels: List of significant levels.
    - replacement_value: Value to replace levels not present in list_levels.

    Returns:
    - df_processed: DataFrame after replacing levels.
    """
    replacement_value='others'
    # Get rows for levels not present in list_levels
    new_df = df[~df[column].isin(list_source)]
    # Replace the values of these levels with the specified replacement_value
    new_df[column] = replacement_value

    # Get rows for levels present in list_levels
    old_df = df[df[column].isin(list_source)]

    # Concatenate the DataFrames
    df_processed = pd.concat([new_df, old_df])

    return df_processed

def replace_for_medium_with_others(df, column):
    """
    Replace levels in the specified column that are not present in list_levels
    with the value 'others'.

    Parameters:
    - df: DataFrame to perform the replacement on.
    - column: Name of the column where levels are to be replaced.
    - list_levels: List of significant levels.

    Returns:
    - df_processed: DataFrame after replacing levels.
    """

    # Get rows for levels not present in list_levels
    new_df = df[~df[column].isin(list_medium)]
    # Replace the values of these levels with 'others'
    new_df[column] = 'others'

    # Get rows for levels present in list_levels
    old_df = df[df[column].isin(list_medium)]

    # Concatenate the DataFrames
    df_processed = pd.concat([new_df, old_df])

    return df_processed
    

###############################################################################
# Define function to map cities to their respective tiers
###############################################################################

def replace_for_platform_with_others(df, column):
    """
    Replace levels in the specified column that are not present in list_levels
    with the value 'others'.

    Parameters:
    - df: DataFrame to perform the replacement on.
    - column: Name of the column where levels are to be replaced.
    - list_levels: List of significant levels.

    Returns:
    - df_processed: DataFrame after replacing levels.
    """

    # Get rows for levels not present in list_levels
    new_df = df[~df[column].isin(list_platform)]
    # Replace the values of these levels with 'others'
    new_df[column] = 'others'

    # Get rows for levels present in list_levels
    old_df = df[df[column].isin(list_platform)]

    # Concatenate the DataFrames
    df_processed = pd.concat([new_df, old_df])

    return df_processed

    
def map_city_tier():
    # Connect to the SQLite database
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)

    # Read the existing data from the 'loaded_data' table
    loaded_data_query = 'SELECT * FROM loaded_data'
    loaded_data = pd.read_sql_query(loaded_data_query, conn)

    # Map cities to their respective tier using the provided mapping
    loaded_data['city_tier'] = loaded_data['city_mapped'].map(city_tier_mapping)
    loaded_data['city_tier'] = loaded_data['city_tier'].fillna(3.0)
    print(loaded_data.columns.values)
    print(loaded_data.shape)
    loaded_data = loaded_data.drop(['city_mapped'], axis = 1, errors='ignore')
    # Save the mapped data to the database
    loaded_data.to_sql('city_tier_mapped', conn, index=False, if_exists='replace')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

###############################################################################
# Define function to map insignificant categorial variables to "others"
###############################################################################


def map_categorical_vars(): 
    # Construct the full path of the database file
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)

    # Read the existing data from the 'city_tier_mapped' table
    city_tier_query = 'SELECT * FROM city_tier_mapped'
    city_tier_mapped = pd.read_sql_query(city_tier_query, conn)

    # Map significant variables in 'first_platform_c'
    city_tier_mapped['first_platform_c'] = city_tier_mapped['first_platform_c'].apply(lambda x: x if x in list_platform else 'others')

    # Map significant variables in 'first_utm_medium_c'
    city_tier_mapped['first_utm_medium_c'] = city_tier_mapped['first_utm_medium_c'].apply(lambda x: x if x in list_medium else 'others')

    # Map significant variables in 'first_utm_source_c'
    city_tier_mapped['first_utm_source_c'] = city_tier_mapped['first_utm_source_c'].apply(lambda x: x if x in list_source else 'others')

    # Save the mapped data to the database
    city_tier_mapped.to_sql('categorical_variables_mapped', conn, index=False, if_exists='replace')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    #print(f'Categorical variables mapping has been successfully applied and saved to the database at {db_file_path}.')



##############################################################################
# Define function that maps interaction columns into 4 types of interactions
##############################################################################
def interactions_mapping():
    # Construct the full path of the database file
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)

    # Read the existing data from the 'categorical_variables_mapped' table
    cat_vars_query = 'SELECT * FROM categorical_variables_mapped'
    cat_vars_mapped = pd.read_sql_query(cat_vars_query, conn)

    # Read the interaction mappings from the CSV file
    interaction_mapping = pd.read_csv(INTERACTION_MAPPING, index_col=[0])
    cat_vars_mapped = melt_dataframe(cat_vars_mapped)
    cat_vars_mapped['interaction_value'] = cat_vars_mapped['interaction_value'].fillna(0)
    # Merge interaction mappings with categorical variables
    interactions_mapped = pd.merge(cat_vars_mapped, interaction_mapping, how='left', on='interaction_type')
    if 'app_complete_flag' not in cat_vars_mapped.columns.values:
        NOT_FEATURES.append('app_complete_flag')
    
    # Pivot the data for inference
    #interactions_mapped_inference = interactions_mapped.pivot_table(index=INDEX_COLUMNS_INFERENCE, columns='interaction_mapping', values='interaction_value', aggfunc='sum').reset_index()

    # Concatenate the training and inference data
    #interactions_mapped_final = pd.concat([interactions_mapped_train, interactions_mapped_inference])
    #print(interactions_mapped.columns.values)
    # Pivot the data for training
    #interactions_mapped_final = pivot_interaction_mapping(interactions_mapped,'interaction_mapping')
    # Drop unnecessary features
    
    # Save the mapped interactions to the database
    interactions_mapped_final = pivot_interaction_mapping(interactions_mapped,'interaction_mapping')
    print(NOT_FEATURES)
    interactions_mapped_final.to_sql('interactions_mapped', conn, index=False, if_exists='replace')
    print('interactions_mapped_final '+interactions_mapped_final.columns.values)
    model_input = interactions_mapped_final.drop(NOT_FEATURES,axis=1)
    # Save the model input data to the database
    model_input.to_sql('model_input', conn, index=False, if_exists='replace')
    print('model_input '+model_input.columns.values)
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    #print(f'Interactions mapping and feature selection completed. Data saved to the database at {db_file_path}.')

######################################################################################################################################################################################################################################################

def pivot_interaction_mapping(df, interaction_mapping_column):
    """
    Pivot the interaction mapping column values to individual columns in the dataset.

    Parameters:
    - df: DataFrame to be pivoted.
    - index_columns: List of columns to be used as index while pivoting.
    - interaction_mapping_column: Column containing interaction mapping values.

    Returns:
    - pivoted_df: DataFrame after pivoting.
    """

    # Ensure that the interaction_mapping_column is a string
    if not isinstance(interaction_mapping_column, str):
        raise ValueError("interaction_mapping_column must be a string")

    # Pivot the data
    pivoted_df = df.pivot_table(
        values='interaction_value', 
        index=INDEX_COLUMNS_TRAINING, 
        columns=interaction_mapping_column, 
        aggfunc='sum'
    ).reset_index()

    return pivoted_df


def melt_dataframe(df, var_name='interaction_type', value_name='interaction_value'):
    """
    Melt the DataFrame based on specified id_vars.

    Parameters:
    - df: DataFrame to be melted.
    - id_vars: List of columns to be retained as identifier variables.
    - var_name: Name of the variable column (default is 'interaction_type').
    - value_name: Name of the value column (default is 'interaction_value').

    Returns:
    - df_melted: Melted DataFrame.
    """
    df_melted = pd.melt(df, id_vars=INDEX_COLUMNS_TRAINING, var_name=var_name, value_name=value_name)
    return df_melted