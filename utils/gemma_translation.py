# utils/gemma_translation.py

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from llama_cpp import Llama
import streamlit as st
from typing import Iterator, Optional, List
import re

from .chunking import chunk_text_with_separators

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.path.join("local_llms", "gemma-3-12b-it-Q4_K_M.gguf")
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_GPU_LAYERS = 48  # Adjust based on available GPU memory
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CHUNK_SIZE = 3000  # Maximum number of tokens per chunk for safety

class LlamaCppTokenizerAdapter:
    """
    Adapter class to make llama-cpp Llama model compatible with chunking utility 
    which expects a HuggingFace tokenizer interface.
    """
    def __init__(self, llama_model):
        self.model = llama_model
        
    def encode(self, text, add_special_tokens=False):
        """
        Tokenize text using llama-cpp's tokenize method.
        
        Args:
            text: Text to tokenize
            add_special_tokens: Ignored (included for compatibility)
            
        Returns:
            List of token IDs
        """
        try:
            return self.model.tokenize(bytes(text, "utf-8"))
        except Exception as e:
            logger.warning(f"Tokenization error: {str(e)}")
            # Fallback to character-based approximate tokenization (4 chars ≈ 1 token)
            return [0] * (len(text) // 4 + 1)

class GemmaTranslator:
    """
    Translator using Gemma 3 model in GGUF format with streaming capability.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model is loaded."""
        if cls._instance is None:
            cls._instance = super(GemmaTranslator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the Gemma translator if not already initialized."""
        if not hasattr(self, 'initialized'):
            self.initialized = False
            self.model_path = Path(MODEL_PATH)
            self.model = None
            self.tokenizer = None
    
    def load_model(self, 
                  n_gpu_layers: int = DEFAULT_GPU_LAYERS, 
                  context_size: int = DEFAULT_CONTEXT_SIZE) -> None:
        """
        Load the Gemma model with specified parameters.
        
        Args:
            n_gpu_layers: Number of layers to offload to GPU
            context_size: Context window size
        """
        if self.initialized:
            return
        
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            logger.info(f"Loading Gemma model from {self.model_path}...")
            
            # Create Llama model with streaming capability
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=context_size,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            # Create tokenizer adapter
            self.tokenizer = LlamaCppTokenizerAdapter(self.model)
            
            self.initialized = True
            logger.info("Gemma model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {str(e)}")
            raise
    
    def generate_translation_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Create a prompt for translation.
        
        Args:
            text: Text to translate
            src_lang: Source language code ('en', 'ru', 'kk')
            tgt_lang: Target language code ('en', 'ru', 'kk')
            
        Returns:
            Formatted prompt for the model
        """
        lang_map = {
            'en': 'English',
            'ru': 'Russian',
            'kk': 'Kazakh'
        }
        
        source_lang = lang_map.get(src_lang, 'Unknown')
        target_lang = lang_map.get(tgt_lang, 'Unknown')
        
        system_prompt = (
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"Provide only the translated text without explanations, introductions, or comments."
        )
        
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{text}\n<|assistant|>\n"
        return prompt
    
    def is_text_too_large(self, text: str) -> bool:
        """
        Check if text is too large for the model's context window.
        
        Args:
            text: Input text
            
        Returns:
            True if text needs chunking, False otherwise
        """
        if not self.initialized:
            self.load_model()
            
        # Use actual tokenization when possible
        try:
            tokens = self.model.tokenize(bytes(text, "utf-8"))
            token_count = len(tokens)
        except Exception:
            # Fallback to character-based approximation
            token_count = len(text) / 4
        
        # Allow for prompt overhead and model's response tokens
        threshold = DEFAULT_CONTEXT_SIZE * 0.9
        
        return token_count > threshold
    
    def _split_text_into_sentences(self, text: str, lang: str) -> List[str]:
        """
        Split text into sentences for simple chunking when full chunking fails.
        
        Args:
            text: Text to split
            lang: Language code
            
        Returns:
            List of sentences
        """
        if lang in ['ru', 'kk']:
            # Russian/Kazakh sentence pattern
            pattern = r'(?<=[.!?])\s+'
        else:
            # English sentence pattern
            pattern = r'(?<=[.!?])\s+'
            
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def translate(self, 
                 text: str, 
                 src_lang: str, 
                 tgt_lang: str,
                 temperature: float = 0.1,
                 top_p: float = 0.95,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """
        Translate text using Gemma model.
        
        Args:
            text: Text to translate
            src_lang: Source language code ('en', 'ru', 'kk')
            tgt_lang: Target language code ('en', 'ru', 'kk')
            temperature: Generation temperature (lower = more deterministic)
            top_p: Top-p sampling threshold
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Translated text
        """
        if not self.initialized:
            self.load_model()
        
        # Check if text is too large and needs chunking
        if self.is_text_too_large(text):
            logger.info("Text is too large, using chunking")
            return self._translate_large_text(text, src_lang, tgt_lang, temperature, top_p, max_tokens)
        
        # Prepare prompt for normal-sized text
        prompt = self.generate_translation_prompt(text, src_lang, tgt_lang)
        
        try:
            # Generate translation
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|user|>", "<|system|>"],
                echo=False
            )
            
            # Extract translated text
            if response and "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["text"].strip()
            else:
                logger.warning("Empty or invalid response from model")
                return ""
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return f"Error: {str(e)}"
    
    def _translate_large_text(self, 
                            text: str, 
                            src_lang: str, 
                            tgt_lang: str,
                            temperature: float = 0.1,
                            top_p: float = 0.95,
                            max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """
        Translate large text by splitting it into chunks.
        
        Args:
            text: Text to translate
            src_lang: Source language code ('en', 'ru', 'kk')
            tgt_lang: Target language code ('en', 'ru', 'kk')
            temperature: Generation temperature
            top_p: Top-p sampling threshold
            max_tokens: Maximum tokens to generate
            
        Returns:
            Translated text with chunks combined
        """
        try:
            # Determine language for chunking
            lang_for_chunking = 'russian' if src_lang in ['ru', 'kk'] else 'english'
            
            # Use the chunking utility to split text
            try:
                chunks_with_seps = chunk_text_with_separators(
                    text=text,
                    tokenizer=self.tokenizer,
                    max_tokens=DEFAULT_CHUNK_SIZE,
                    lang=lang_for_chunking
                )
            except Exception as chunk_error:
                # Fallback to simpler sentence splitting if advanced chunking fails
                logger.warning(f"Advanced chunking failed: {str(chunk_error)}. Using simple sentence splitting.")
                sentences = self._split_text_into_sentences(text, src_lang)
                chunks_with_seps = [(sent, " ") for sent in sentences]
            
            translations = []
            for chunk_idx, (chunk, separator) in enumerate(chunks_with_seps):
                if not chunk.strip():
                    translations.append(separator)
                    continue
                
                logger.info(f"Translating chunk {chunk_idx + 1} of {len(chunks_with_seps)}")
                
                # Translate each chunk
                prompt = self.generate_translation_prompt(chunk, src_lang, tgt_lang)
                try:
                    response = self.model(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=["<|user|>", "<|system|>"],
                        echo=False
                    )
                    
                    if response and "choices" in response and len(response["choices"]) > 0:
                        translated_chunk = response["choices"][0]["text"].strip()
                        translations.append(translated_chunk)
                        translations.append(separator)
                    else:
                        logger.warning(f"Empty response for chunk {chunk_idx}")
                        translations.append(f"[Translation error]")
                        translations.append(separator)
                        
                except Exception as e:
                    logger.error(f"Error translating chunk {chunk_idx}: {str(e)}")
                    translations.append(f"[Error: {str(e)}]")
                    translations.append(separator)
            
            # Combine all translated chunks
            combined_text = ''.join(translations)
            
            # Cleanup and postprocessing
            return self._postprocess_translation(combined_text)
            
        except Exception as e:
            logger.error(f"Large text translation error: {str(e)}")
            return f"Error: {str(e)}"
    
    def _postprocess_translation(self, text: str) -> str:
        """Clean up and format the translated text."""
        # Remove multiple spaces
        text = ' '.join(text.split())
        # Fix punctuation spacing
        text = text.replace(' .', '.').replace(' ,', ',')
        text = text.replace(' !', '!').replace(' ?', '?')
        # Fix quote spacing
        text = text.replace('" ', '"').replace(' "', '"')
        return text
    
    def translate_streaming(self, 
                           text: str, 
                           src_lang: str, 
                           tgt_lang: str,
                           temperature: float = 0.1,
                           top_p: float = 0.95,
                           max_tokens: int = DEFAULT_MAX_TOKENS) -> Iterator[str]:
        """
        Stream translation using Gemma model.
        
        Args:
            text: Text to translate
            src_lang: Source language code ('en', 'ru', 'kk')
            tgt_lang: Target language code ('en', 'ru', 'kk')
            temperature: Generation temperature (lower = more deterministic)
            top_p: Top-p sampling threshold
            max_tokens: Maximum number of tokens to generate
            
        Yields:
            Chunks of translated text as they're generated
        """
        if not self.initialized:
            self.load_model()
        
        # Check if text is too large and needs chunking
        if self.is_text_too_large(text):
            logger.info("Text is too large, using chunked streaming")
            yield from self._translate_large_text_streaming(text, src_lang, tgt_lang, temperature, top_p, max_tokens)
            return
            
        # Prepare prompt for normal-sized text
        prompt = self.generate_translation_prompt(text, src_lang, tgt_lang)
        
        try:
            # Stream translation
            for chunk in self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|user|>", "<|system|>"],
                echo=False,
                stream=True
            ):
                if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                    token = chunk["choices"][0]["text"]
                    if token:
                        yield token
                        
        except Exception as e:
            logger.error(f"Streaming translation error: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _translate_large_text_streaming(self, 
                                      text: str, 
                                      src_lang: str, 
                                      tgt_lang: str,
                                      temperature: float = 0.1,
                                      top_p: float = 0.95,
                                      max_tokens: int = DEFAULT_MAX_TOKENS) -> Iterator[str]:
        """
        Stream translation of large text by chunks.
        
        Args:
            text: Text to translate
            src_lang: Source language code ('en', 'ru', 'kk')
            tgt_lang: Target language code ('en', 'ru', 'kk')
            temperature: Generation temperature
            top_p: Top-p sampling threshold
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of translated text
        """
        try:
            # Determine language for chunking
            lang_for_chunking = 'russian' if src_lang in ['ru', 'kk'] else 'english'
            
            # Use the chunking utility to split text
            try:
                chunks_with_seps = chunk_text_with_separators(
                    text=text,
                    tokenizer=self.tokenizer,
                    max_tokens=DEFAULT_CHUNK_SIZE,
                    lang=lang_for_chunking
                )
            except Exception as chunk_error:
                # Fallback to simpler sentence splitting if advanced chunking fails
                logger.warning(f"Advanced chunking failed: {str(chunk_error)}. Using simple sentence splitting.")
                sentences = self._split_text_into_sentences(text, src_lang)
                chunks_with_seps = [(sent, " ") for sent in sentences]
            
            for chunk_idx, (chunk, separator) in enumerate(chunks_with_seps):
                if not chunk.strip():
                    yield separator
                    continue
                
                if chunk_idx > 0:
                    yield "\n\n"  # Add visual separation between chunks
                
                # Translate each chunk
                prompt = self.generate_translation_prompt(chunk, src_lang, tgt_lang)
                
                try:
                    # Stream chunk translation
                    for token_chunk in self.model(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=["<|user|>", "<|system|>"],
                        echo=False,
                        stream=True
                    ):
                        if token_chunk and "choices" in token_chunk and len(token_chunk["choices"]) > 0:
                            token = token_chunk["choices"][0]["text"]
                            if token:
                                yield token
                    
                    # Add separator after chunk
                    yield separator
                    
                except Exception as e:
                    logger.error(f"Error streaming chunk {chunk_idx}: {str(e)}")
                    yield f"\n[Error translating part {chunk_idx + 1}: {str(e)}]\n"
            
        except Exception as e:
            logger.error(f"Large text streaming error: {str(e)}")
            yield f"\nError: {str(e)}"


def gemma_translate(text: str, src_lang: str, tgt_lang: str, streaming: bool = True) -> Optional[Iterator[str]]:
    """
    Main function to translate text using Gemma 3 model.
    
    Args:
        text: Text to translate
        src_lang: Source language code ('en', 'ru', 'kk')
        tgt_lang: Target language code ('en', 'ru', 'kk')
        streaming: Whether to stream the output
        
    Returns:
        If streaming is True: Iterator yielding chunks of translated text
        If streaming is False: Complete translated text
    """
    if not text or not src_lang or not tgt_lang:
        return "" if not streaming else iter([""])
    
    translator = GemmaTranslator()
    
    try:
        if streaming:
            return translator.translate_streaming(text, src_lang, tgt_lang)
        else:
            return translator.translate(text, src_lang, tgt_lang)
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return "" if not streaming else iter([f"Error: {str(e)}"])


def display_streaming_translation(text: str, src_lang: str, tgt_lang: str) -> tuple:
    """
    Display streaming translation in a Streamlit app.
    
    Args:
        text: Text to translate
        src_lang: Source language code ('en', 'ru', 'kk')
        tgt_lang: Target language code ('en', 'ru', 'kk')
        
    Returns:
        tuple: (translated_text, needs_chunking)
    """
    if not text:
        return "", False
    
    # Check if text needs chunking
    translator = GemmaTranslator()
    if not translator.initialized:
        translator.load_model()
    needs_chunking = translator.is_text_too_large(text)
    
    # Create placeholder for streaming output
    placeholder = st.empty()
    result = ""
    
    # Stream translation
    for token in gemma_translate(text, src_lang, tgt_lang, streaming=True):
        result += token
        placeholder.markdown(result)
    
    return result, needs_chunking 