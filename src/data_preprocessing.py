import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')

for resource in ['corpora/stopwords', 'tokenizers/punkt']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[1], quiet=True)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(LOG_DIR, 'data_preprocessing.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
        test_data = pd.read_csv(os.path.join(RAW_DIR, 'test.csv'))
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/interim
        train_processed_data.to_csv(os.path.join(INTERIM_DIR, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(INTERIM_DIR, 'test_processed.csv'), index=False)
        
        logger.debug('Processed data saved to %s', INTERIM_DIR)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        print(f'Error: {e}')
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
        print(f'Error: {e}')
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()