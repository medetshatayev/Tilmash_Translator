# readability_indices.py

from nltk.tokenize import sent_tokenize, word_tokenize
import pyphen
import re
from IPython.display import display, HTML

def count_syllables(word, lang):
    if lang == 'kk':
        # Используем простой алгоритм для казахского языка
        word = word.lower()
        vowels = "аеёиоуыэюяіүұөө"
        syllables = sum(1 for char in word if char in vowels)
        return max(1, syllables)
    else:
        # Для русского и английского используем Pyphen
        dic = pyphen.Pyphen(lang=lang)
        hyphens = dic.inserted(word)
        return max(1, hyphens.count('-') + 1)

# Функции для определения сложных слов
def is_complex_word(word, lang, syllable_threshold=3):
    syllables = count_syllables(word, lang)
    return syllables >= syllable_threshold

# Функции для расчёта индексов удобочитаемости
def flesch_reading_ease(text, lang):
    sentences = sent_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = word_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = [word for word in words if word.isalpha()]
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    syllable_count = sum([count_syllables(word, lang) for word in words])
    asl = num_words / num_sentences  # Средняя длина предложения
    asw = syllable_count / num_words  # Среднее количество слогов в слове
    if lang == 'ru':
        fre = 206.835 - (1.3 * asl) - (60.1 * asw)
    elif lang == 'en':
        fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    elif lang == 'kk':
        # Предположительные коэффициенты для казахского языка
        fre = 206.835 - (1.2 * asl) - (70 * asw)
    else:
        fre = 0
    return fre

def flesch_kincaid_grade_level(text, lang):
    sentences = sent_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = word_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = [word for word in words if word.isalpha()]
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    syllable_count = sum([count_syllables(word, lang) for word in words])
    asl = num_words / num_sentences
    asw = syllable_count / num_words
    if lang == 'ru':
        fkgl = (0.5 * asl) + (8.4 * asw) - 15.59
    elif lang == 'en':
        fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
    elif lang == 'kk':
        fkgl = (0.5 * asl) + (9 * asw) - 13
    else:
        fkgl = 0
    return fkgl

def gunning_fog_index(text, lang):
    sentences = sent_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = word_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = [word for word in words if word.isalpha()]
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    complex_words = [word for word in words if is_complex_word(word, lang)]
    percentage_complex = (len(complex_words) / num_words) * 100
    asl = num_words / num_sentences
    fog_index = 0.4 * (asl + percentage_complex)
    return fog_index

def smog_index(text, lang):
    sentences = sent_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = word_tokenize(text, language='russian' if lang == 'ru' else 'english')
    words = [word for word in words if word.isalpha()]
    num_sentences = len(sentences)
    complex_words = [word for word in words if is_complex_word(word, lang)]
    num_complex = len(complex_words)
    if num_sentences >= 3:
        smog = 1.0430 * ((num_complex * (30 / num_sentences)) ** 0.5) + 3.1291
    else:
        smog = 0
    return smog

# Функция для выделения сложных слов и предложений
def highlight_complex_text(text, lang):
    sentences = sent_tokenize(text, language='russian' if lang == 'ru' else 'english')
    highlighted_sentences = []
    complex_words_list = []
    for sentence in sentences:
        words = word_tokenize(sentence, language='russian' if lang == 'ru' else 'english')
        words_filtered = [word for word in words if word.isalpha()]
        complex_words = [word for word in words_filtered if is_complex_word(word, lang)]
        complex_words_list.extend(complex_words)
        if len(words_filtered) > 0 and (len(complex_words) / len(words_filtered)) > 0.3:
            highlighted_sentence = f"<mark>{sentence}</mark>"
        else:
            highlighted_sentence = sentence
            for word in complex_words:
                highlighted_sentence = re.sub(r'\b{}\b'.format(re.escape(word)), f"<b>{word}</b>", highlighted_sentence)
        highlighted_sentences.append(highlighted_sentence)
    highlighted_text = ' '.join(highlighted_sentences)
    return highlighted_text, complex_words_list

# Основная функция
def analyze_text(text, lang_code):
    if lang_code not in ['ru', 'en', 'kk']:
        print('Unsupported language code. Please use "ru" for Russian, "en" for English, or "kk" for Kazakh.')
        return
    fre = flesch_reading_ease(text, lang_code)
    fkgl = flesch_kincaid_grade_level(text, lang_code)
    fog = gunning_fog_index(text, lang_code)
    smog = smog_index(text, lang_code)

    highlighted_text, complex_words = highlight_complex_text(text, lang_code)

    # Вывод результатов
    print(f"Язык: {'Русский' if lang_code == 'ru' else 'Английский' if lang_code == 'en' else 'Казахский'}")
    print(f"Индекс удобочитаемости Флеша: {fre:.2f}")
    print(f"Индекс Флеша-Кинкейда: {fkgl:.2f}")
    print(f"Индекс тумана Ганнинга: {fog:.2f}")
    print(f"Индекс SMOG: {smog:.2f}")
    print("\nСложные слова:")
    print(', '.join(set(complex_words)))
    print("\nТекст с выделениями:")
    display(HTML(highlighted_text))