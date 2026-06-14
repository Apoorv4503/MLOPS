import pandas as pd
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
##import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    log_file_path = LOG_DIR / 'feature_engineering.log'
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

"""def load_params(params_path: str) -> dict:
    
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise """

def resolve_path(file_path: str | Path) -> Path:
    """Resolve a file path relative to the project base directory."""
    file_path = Path(file_path)
    return file_path if file_path.is_absolute() else BASE_DIR / file_path


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    path = resolve_path(file_path)
    try:
        df = pd.read_csv(path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TfIdf to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str | Path) -> None:
    """Save the dataframe to a CSV file."""
    path = resolve_path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug('Data saved to %s', path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        # max_features = params['feature_engineering']['max_features']
        max_features = 50

        train_data = load_data(BASE_DIR / 'data' / 'interim' / 'train_processed.csv')
        test_data = load_data(BASE_DIR / 'data' / 'interim' / 'test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, BASE_DIR / 'data' / 'processed' / 'train_tfidf.csv')
        save_data(test_df, BASE_DIR / 'data' / 'processed' / 'test_tfidf.csv')
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()