DB_PATH = '/home/Assignment/02_training_pipeline/scripts/'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db'
ML_DB_PATH = '/home/Assignment/database/'
MLFLOW_DB_NAME = 'Lead_scoring_mlflow_production.db'
DATA_DIR = '/home/Assignment/02_training_pipeline/notebooks/data'
CLEANED_DATA = 'cleaned_data.csv'
#DB_FILE_MLFLOW = 
TRACKING_URI = 'http://0.0.0.0:6008'
EXPERIMENT = 'Lead_scoring_mlflow_production'


# model config imported from pycaret experimentation
# model_config = 

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES = ['city_tier_2.0' ,'city_tier_3.0', 'first_platform_c_Level3', 'first_platform_c_Level1','first_platform_c_others', 'first_platform_c_Level7','first_platform_c_Level2','first_platform_c_Level8', 'first_utm_medium_c_Level0','first_utm_medium_c_Level3','first_utm_medium_c_Level2', 'first_utm_medium_c_others','first_utm_medium_c_Level33','first_utm_medium_c_Level6', 'first_utm_medium_c_Level30','first_utm_medium_c_Level26','first_utm_medium_c_Level5', 'first_utm_medium_c_Level16','first_utm_medium_c_Level13','first_utm_medium_c_Level9', 'first_utm_medium_c_Level20','first_utm_medium_c_Level8','first_utm_medium_c_Level15', 'first_utm_medium_c_Level10','first_utm_medium_c_Level43','first_utm_medium_c_Level4', 'first_utm_source_c_others','first_utm_source_c_Level0','first_utm_source_c_Level16', 'first_utm_source_c_Level7','first_utm_source_c_Level5','first_utm_source_c_Level6', 'first_utm_source_c_Level4','first_utm_source_c_Level14','referred_lead_1.0']
# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = ['city_tier','first_platform_c','first_utm_medium_c','first_utm_source_c','referred_lead']
