# utils/text_processing.py

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def detect_language(text):
    try:
        lang = detect(text)
        # Convert 'kk' from langdetect if it indeed returns 'kk' for Kazakh
        if lang not in ['ru', 'en', 'kk']:
            return None
        return lang
    except:
        return None