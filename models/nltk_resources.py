# models/nltk_resources.py

import nltk
import logging

def setup_nltk():
    nltk_data_dir = 'nltk_data'

    # Add the nltk_data directory to the NLTK data path
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    # Define the required package
    required_package = 'punkt_tab'

    # Check if the package is installed locally
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logging.info(f"Downloading NLTK package: {required_package}")
        nltk.download(required_package, download_dir=nltk_data_dir, quiet=True)