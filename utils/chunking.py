# utils/chunking.py

import logging
from pysbd import Segmenter
import re


def chunk_text_with_separators(text, tokenizer, max_tokens, lang):
    """
    Splits the input text into chunks with preserved separators, optimized for handling lists and tables.

    Args:
        text (str): The input text to be chunked.
        tokenizer: Tokenizer object used to encode text into tokens.
        max_tokens (int): Maximum number of tokens allowed per chunk.
        lang (str): Language of the text, used for sentence segmentation.

    Returns:
        list: A list of tuples, each containing a chunk of text and its corresponding separator.
    """
    # Split text into sentences while preserving separators
    sentences_with_seps = _split_technical_sentences(text, lang)
    chunks = []
    current_chunk = []
    current_length = 0
    current_separators = []

    for sentence, sep in sentences_with_seps:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_len = len(sentence_tokens)

        if sentence_len == 0:
            continue

        # Handle special cases like lists and tables
        if _is_list_item(sentence) or _is_table_header(sentence):
            if current_chunk:
                # Finalize the current chunk before processing special items
                chunks.append((' '.join(current_chunk), ''.join(current_separators)))
                current_chunk = []
                current_length = 0
                current_separators = []

            # Process list items as separate chunks
            chunks.extend(_process_special_item(sentence, sep, tokenizer, max_tokens))
            continue

        # Add sentence to the current chunk if it fits
        if current_length + sentence_len <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_len
            current_separators.append(sep)
        else:
            # Finalize the current chunk and start a new one
            if current_chunk:
                chunks.append((' '.join(current_chunk), ''.join(current_separators)))
            current_chunk = [sentence]
            current_length = sentence_len
            current_separators = [sep]

    # Add any remaining text to the final chunk
    if current_chunk:
        chunks.append((' '.join(current_chunk), ''.join(current_separators)))

    return chunks


def _split_technical_sentences(text, lang):
    """Enhanced splitting for technical documents with lists and tables"""
    # Handle numbered lists and bullet points
    text = re.sub(r'(\n\s*\d+\.)', r'\n§§§\1', text)
    # Handle colon-terminated headers
    text = re.sub(r'(:\s*\n)', r'\1§§§', text)

    sentences = []
    separators = []

    if lang == 'russian':
        segmenter = Segmenter(language='ru', clean=False)
        raw_sentences = segmenter.segment(text)
    else:
        raw_sentences = re.split(r'([.!?])(\s*)', text)

    buffer = ''
    current_sep = ''

    for part in raw_sentences:
        if '§§§' in part:
            parts = part.split('§§§')
            for p in parts[:-1]:
                if p.strip():
                    sentences.append(p.strip())
                    separators.append(current_sep)
                current_sep = ''
            buffer = parts[-1]
        else:
            buffer += part

        # Process buffer when we hit sentence boundaries
        if lang == 'russian':
            if buffer.strip() and any(buffer.endswith(c) for c in ['.', '!', '?', ':']):
                sentences.append(buffer.strip())
                separators.append(current_sep)
                buffer = ''
                current_sep = ''
        else:
            if re.search(r'[.!?:]$', buffer):
                sentences.append(buffer.strip())
                separators.append(current_sep)
                buffer = ''
                current_sep = ''

    if buffer.strip():
        sentences.append(buffer.strip())
        separators.append(current_sep)

    return list(zip(sentences, separators))


def _is_list_item(text):
    return re.match(r'^\s*(\d+\.|\-|\*)\s', text)


def _is_table_header(text):
    return re.search(r':\s*$', text) and re.search(r'[A-ZА-Я]{3,}', text)


def _process_special_item(text, separator, tokenizer, max_tokens):
    """Process list items and table headers as atomic units"""
    chunks = []
    current_chunk = []
    current_length = 0

    sentences = re.split(r'(\n+)', text)
    for sentence in sentences:
        if not sentence.strip():
            continue

        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_count = len(tokens)

        if token_count > max_tokens:
            # Handle oversized items with careful splitting
            parts = re.split(r'([,;])', sentence)
            for part in parts:
                if not part.strip():
                    continue
                part_tokens = tokenizer.encode(part, add_special_tokens=False)
                part_len = len(part_tokens)

                if current_length + part_len > max_tokens:
                    chunks.append((' '.join(current_chunk), separator))
                    current_chunk = [part]
                    current_length = part_len
                else:
                    current_chunk.append(part)
                    current_length += part_len
        else:
            if current_length + token_count > max_tokens:
                chunks.append((' '.join(current_chunk), separator))
                current_chunk = [sentence]
                current_length = token_count
            else:
                current_chunk.append(sentence)
                current_length += token_count

    if current_chunk:
        chunks.append((' '.join(current_chunk), separator))

    return chunks