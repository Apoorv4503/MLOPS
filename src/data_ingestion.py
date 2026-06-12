import os
import logging
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(LOG_DIR, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> Dict:
    """Load parameters from a YAML file."""
    if not os.path.exists(params_path):
        logger.warning('Parameters file not found, using default values: %s', params_path)
        return {}

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file) or {}
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path=os.path.join(ROOT_DIR, 'params.yaml'))
        test_size = params.get('data_ingestion', {}).get('test_size', 0.2)

        local_data_path = os.path.join(ROOT_DIR, 'spam.csv')
        if os.path.exists(local_data_path):
            data_source = local_data_path
        else:
            data_source = 'https://raw.githubusercontent.com/Apoorv4503/MLOPS/refs/heads/main/spam.csv'

        df = load_data(data_url=data_source)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, DATA_DIR)
        logger.info('Data ingestion completed successfully')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()