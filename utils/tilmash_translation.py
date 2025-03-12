# utils/tilmash_translation.py

import logging
import re
import os
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TranslationPipeline
from .chunking import chunk_text_with_separators
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    logging.warning("HF_TOKEN not found in environment variables. Model downloading might fail.")
else:
    login(token=hf_token)


def tilmash_translate(input_text, src_lang, tgt_lang, model=None, tokenizer=None, max_length=512):
    """Main translation function with structure preservation"""

    lang_map = {
        'ru': 'rus_Cyrl',
        'en': 'eng_Latn',
        'kk': 'kaz_Cyrl'
    }

    # Validate language pair
    if src_lang not in lang_map or tgt_lang not in lang_map:
        logging.error(f"Unsupported language pair: {src_lang} -> {tgt_lang}")
        return ""

    # Initialize model and tokenizer with error handling
    if model is None or tokenizer is None:
        model_name = "issai/tilmash"
        cache_dir = "local_llms"
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # First try to load the model locally
            logging.info("Trying to load Tilmash model from local cache...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=True
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=True
                )
                logging.info("Successfully loaded model from local cache.")
            except OSError:
                # If local loading fails, download the model
                logging.info("Model not found locally. Downloading from Hugging Face...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                logging.info("Successfully downloaded and loaded the model.")
        except ValueError as e:
            logging.error(f"Invalid model configuration: {str(e)}")
            return ""
        except Exception as e:
            logging.error(f"Unexpected error during model initialization: {str(e)}")
            return ""

    # Configure translation pipeline with optimized parameters
    pipeline_tilmash = TranslationPipeline(
        model=model,
        tokenizer=tokenizer,
        src_lang=lang_map[src_lang],
        tgt_lang=lang_map[tgt_lang],
        max_length=max_length,
        num_beams=7,  # Increased for better exploration
        early_stopping=True,
        repetition_penalty=1.3,  # Adjusted to reduce repetition
        no_repeat_ngram_size=2,
        length_penalty=1.1,  # Slightly encourages longer translations
        truncation=True,
        clean_up_tokenization_spaces=True
    )

    # Tokenize input for length check
    token_ids = tokenizer.encode(input_text, add_special_tokens=False)
    total_len = len(token_ids)

    # Translate sentence by sentence
    sentences = re.split(r'(?<=[.!?]) +', input_text)  # Split text into sentences
    translated_sentences = []

    for sentence in sentences:
        if sentence.strip():
            result = pipeline_tilmash(sentence)
            translated_sentence = _extract_translation(result)
            translated_sentences.append(translated_sentence)

    return ' '.join(translated_sentences)


def _extract_translation(result):
    """Safe extraction of translation text from pipeline output"""
    try:
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('translation_text', '').strip()
        return ""
    except Exception as e:
        logging.error(f"Translation extraction error: {str(e)}")
        return ""


def _process_large_text(text, src_lang, pipeline, tokenizer, max_length):
    """Process long documents with structure preservation"""
    try:
        chunks_with_seps = chunk_text_with_separators(
            text=text,
            tokenizer=tokenizer,
            max_tokens=int(0.9 * max_length),
            lang='russian' if src_lang in ['ru', 'kk'] else 'english'
        )
    except Exception as e:
        logging.error(f"Chunking failed: {str(e)}")
        return ""

    translations = []
    prev_separator = None

    for chunk_idx, (chunk, separator) in enumerate(chunks_with_seps):
        if not chunk.strip():
            translations.append(separator)
            continue

        try:
            # Process chunk through translation pipeline
            result = pipeline(chunk)
            translated = _extract_translation(result)

            # Preserve original document structure
            if prev_separator:
                translations.append(prev_separator)

            # Add indentation for list items and tables
            if _is_structured_element(chunk):
                translated = _preserve_structure(translated, chunk)

            translations.append(translated)
            prev_separator = separator

        except Exception as e:
            logging.error(f"Chunk {chunk_idx + 1} error: {str(e)}")
            translations.append(f"<<ERROR: {chunk[:50]}...>>{separator or ' '}")
            prev_separator = separator

    # Assemble final text with cleanup
    final_text = ''.join(translations).strip()
    return _postprocess_translation(final_text)


def _is_structured_element(text):
    """Check if text contains document structure elements"""
    return any([
        re.match(r'^\s*(\d+\.|\-|\*)\s', text),  # List items
        re.search(r':\s*$', text) and re.search(r'[A-ZА-Я]{3,}', text),  # Headers
        re.search(r'\|.+\|', text),  # Tables
        re.search(r'\b(Таблица|Table)\b', text, re.IGNORECASE)  # Table labels
    ])


def _preserve_structure(translated, original):
    """Maintain original formatting in translated structured elements"""
    # Preserve list indentation
    if re.match(r'^\s*(\d+\.|\-|\*)\s', original):
        return '\n' + translated.lstrip()

    # Preserve table formatting
    if '|' in original:
        return translated.replace(' | ', '|').replace('| ', '|').replace(' |', '|')

    return translated


def _postprocess_translation(text):
    """Final cleanup of translated text"""
    # Fix list numbering
    text = re.sub(r'\n(\d+)\.\s*\n', r'\n\1. ', text)
    # Repair table formatting
    text = re.sub(r'(:\s*)\n(\S)', r'\1\2', text)
    # Normalize whitespace
    text = re.sub(r'([,:;])\s+', r'\1 ', text)
    text = re.sub(r'\s+([.!?])', r'\1', text)
    # Restore special characters
    text = text.replace('«', '"').replace('»', '"')
    return text