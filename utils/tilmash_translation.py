# utils/tilmash_translation.py

import logging
import re
import os
import threading
import time
import uuid
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TranslationPipeline
from .chunking import chunk_text_with_separators
from huggingface_hub import login
from typing import Iterator
from config import DEFAULT_CONFIG

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    logging.warning("HF_TOKEN not found in environment variables. Model downloading might fail.")
else:
    login(token=hf_token)

# Global tilmash lock file
LOCK_DIR = os.path.join("local_llms", "locks")
os.makedirs(LOCK_DIR, exist_ok=True)
TILMASH_LOCK_FILE = os.path.join(LOCK_DIR, "tilmash.lock")

# Get session timeout from config
SESSION_TIMEOUT = DEFAULT_CONFIG["SESSION_TIMEOUT"]

class ExclusiveResourceLock:
    """File-based lock for exclusive GPU resource access across processes."""
    
    def __init__(self, lock_file, timeout=SESSION_TIMEOUT):
        self.lock_file = lock_file
        self.timeout = timeout
        self.lock_id = str(uuid.uuid4())
        self.acquired = False
        
    def acquire(self):
        """Acquire exclusive lock with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Try to create the lock file
                if not os.path.exists(self.lock_file):
                    with open(self.lock_file, 'w') as f:
                        f.write(f"{self.lock_id}\n{os.getpid()}\n{time.time()}")
                    
                    # Verify we got the lock
                    with open(self.lock_file, 'r') as f:
                        content = f.read().split('\n')
                        if content and content[0] == self.lock_id:
                            self.acquired = True
                            return True
                            
                # Check if lock file is stale (older than 5 minutes)
                elif os.path.exists(self.lock_file):
                    lock_time = os.path.getmtime(self.lock_file)
                    if time.time() - lock_time > 300:  # 5 minutes
                        try:
                            # Remove stale lock
                            os.remove(self.lock_file)
                            continue
                        except:
                            pass
                
                # Wait before retrying
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Lock acquisition error: {str(e)}")
                time.sleep(1)
                
        return False
        
    def release(self):
        """Release the lock if we own it."""
        if not self.acquired:
            return
            
        try:
            if os.path.exists(self.lock_file):
                with open(self.lock_file, 'r') as f:
                    content = f.read().split('\n')
                    if content and content[0] == self.lock_id:
                        os.remove(self.lock_file)
                        self.acquired = False
        except Exception as e:
            logging.error(f"Lock release error: {str(e)}")
            
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class TilmashTranslator:
    """
    Thread-safe translator using Tilmash model
    """
    
    def __init__(self):
        """Initialize the Tilmash translator."""
        # Use thread-local lock
        self._lock = threading.RLock()
        self.initialized = False
        self.model = None
        self.tokenizer = None
        
        # Get session ID
        import streamlit as st
        self.session_id = getattr(st.session_state, 'session_id', str(uuid.uuid4()))
        
    def load_model(self):
        """Load the Tilmash model if not already loaded."""
        with self._lock:
            if self.initialized:
                return self.model, self.tokenizer
                
            try:
                model_name = "issai/tilmash"
                cache_dir = "local_llms"
                
                # Ensure cache directory exists
                os.makedirs(cache_dir, exist_ok=True)
                
                try:
                    # First try to load the model locally
                    logging.info(f"Loading Tilmash model for session {self.session_id[:8]}...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            local_files_only=True
                        )
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            local_files_only=True
                        )
                        logging.info("Successfully loaded model from local cache.")
                    except OSError:
                        # If local loading fails, download the model
                        logging.info("Model not found locally. Downloading from Hugging Face...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            local_files_only=False
                        )
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            local_files_only=False
                        )
                        logging.info("Successfully downloaded and loaded the model.")
                    
                    self.initialized = True
                    return self.model, self.tokenizer
                    
                except ValueError as e:
                    logging.error(f"Invalid model configuration: {str(e)}")
                    raise ValueError(f"Failed to load model: {str(e)}")
                except Exception as e:
                    logging.error(f"Unexpected error during model initialization: {str(e)}")
                    raise Exception(f"Failed to load model: {str(e)}")
            except Exception as e:
                logging.error(f"Failed to load Tilmash model: {str(e)}")
                raise
    
    def unload_model(self):
        """Unload the model to free memory"""
        with self._lock:
            if self.initialized:
                logging.info("Unloading Tilmash model to free memory...")
                self.model = None
                self.tokenizer = None
                self.initialized = False
                
                # Force garbage collection
                import gc
                gc.collect()
                logging.info("Tilmash model unloaded")
    
    def create_pipeline(self, src_lang, tgt_lang, max_length=512):
        """Create a translation pipeline with the loaded model."""
        with self._lock:
            lang_map = {
                'ru': 'rus_Cyrl',
                'en': 'eng_Latn',
                'kk': 'kaz_Cyrl'
            }
            
            # Validate language pair
            if src_lang not in lang_map or tgt_lang not in lang_map:
                raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")
            
            # Make sure model is loaded
            if not self.initialized:
                self.load_model()
            
            # Configure translation pipeline with optimized parameters
            pipeline = TranslationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                src_lang=lang_map[src_lang],
                tgt_lang=lang_map[tgt_lang],
                max_length=max_length,
                num_beams=7,
                early_stopping=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=2,
                length_penalty=1.1,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            
            return pipeline
    
    def translate(self, text, src_lang, tgt_lang, max_length=512):
        """Translate text using the Tilmash model."""
        with self._lock:
            try:
                pipeline = self.create_pipeline(src_lang, tgt_lang, max_length)
                
                # Split text into sentences for better quality
                sentences = re.split(r'(?<=[.!?]) +', text)
                translated_sentences = []
                
                for sentence in sentences:
                    if sentence.strip():
                        result = pipeline(sentence)
                        translated_sentence = _extract_translation(result)
                        translated_sentences.append(translated_sentence)
                
                return ' '.join(translated_sentences)
            except Exception as e:
                logging.error(f"Translation error: {str(e)}")
                return f"Error: {str(e)}"
    
    def translate_streaming(self, text, src_lang, tgt_lang, max_length=512) -> Iterator[str]:
        """Stream translation results sentence by sentence."""
        try:
            # Make sure model is loaded - must be done in the locked section
            with self._lock:
                if not self.initialized:
                    self.load_model()
                pipeline = self.create_pipeline(src_lang, tgt_lang, max_length)
            
            # Check if text is too large for single processing
            # Improved text size detection - check by paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            is_large_text = len(paragraphs) > 3 or len(text) > 1000  # Multiple paragraphs or long text
            
            if is_large_text:
                # Process paragraph by paragraph for structured documents
                for i, paragraph in enumerate(paragraphs):
                    if not paragraph.strip():
                        yield "\n\n"
                        continue
                        
                    # If paragraph itself is too large, process it sentence by sentence
                    if len(paragraph) > 800:
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        for sentence in sentences:
                            if not sentence.strip():
                                continue
                                
                            try:
                                # Only lock the actual model inference
                                with self._lock:
                                    result = pipeline(sentence)
                                translated = _extract_translation(result)
                                yield translated + " "
                            except Exception as e:
                                logging.error(f"Error translating sentence: {str(e)}")
                                yield f"[Error: {str(e)}] "
                    else:
                        # Process whole paragraph at once
                        try:
                            # Only lock the actual model inference
                            with self._lock:
                                result = pipeline(paragraph)
                            translated = _extract_translation(result)
                            yield translated
                            # Add paragraph break after each paragraph
                            if i < len(paragraphs) - 1:
                                yield "\n\n"
                        except Exception as e:
                            logging.error(f"Error translating paragraph: {str(e)}")
                            yield f"[Error translating paragraph: {str(e)}]\n\n"
            else:
                # For short texts, process the entire text at once
                try:
                    # Only lock the actual model inference
                    with self._lock:
                        result = pipeline(text)
                    translated = _extract_translation(result)
                    yield translated
                except Exception as e:
                    logging.error(f"Error translating text: {str(e)}")
                    yield f"[Error: {str(e)}]"
        except Exception as e:
            logging.error(f"Streaming translation error: {str(e)}")
            yield f"Error initializing translation: {str(e)}"


def tilmash_translate(input_text, src_lang, tgt_lang, max_length=512):
    """Main translation function with structure preservation"""
    try:
        translator = TilmashTranslator()
        return translator.translate(input_text, src_lang, tgt_lang, max_length)
    except Exception as e:
        logging.error(f"Translation failed: {str(e)}")
        return f"Translation error: {str(e)}"


def tilmash_translate_streaming(input_text, src_lang, tgt_lang, max_length=512) -> Iterator[str]:
    """Streaming version of the translation function that yields translated sentences one by one"""
    try:
        translator = TilmashTranslator()
        yield from translator.translate_streaming(input_text, src_lang, tgt_lang, max_length)
    except Exception as e:
        logging.error(f"Streaming translation failed: {str(e)}")
        yield f"Translation error: {str(e)}"


def display_tilmash_streaming_translation(text: str, src_lang: str, tgt_lang: str) -> tuple:
    """
    Display streaming translation in a Streamlit app.
    
    Args:
        text: Text to translate
        src_lang: Source language code ('en', 'ru', 'kk')
        tgt_lang: Target language code ('en', 'ru', 'kk')
        
    Returns:
        tuple: (translated_text, needs_chunking)
    """
    import streamlit as st
    
    if not text:
        return "", False
    
    # Check if text needs chunking
    needs_chunking = len(text) > 1000  # Roughly 250 tokens
    
    # Create placeholder for streaming output
    placeholder = st.empty()
    result = ""
    
    # Stream translation
    for sentence in tilmash_translate_streaming(text, src_lang, tgt_lang):
        result += sentence
        placeholder.markdown(result)
    
    return result, needs_chunking


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