# main.py

import streamlit as st
import logging

st.set_page_config(page_title="Translator & Readability", layout="wide")
logging.basicConfig(level=logging.INFO)

import os
import tempfile
import io
from docx import Document

from models.nltk_resources import setup_nltk
from utils.file_readers import read_file
from utils.text_processing import detect_language
from utils.readability_indices import (
    flesch_reading_ease,
    flesch_kincaid_grade_level,
    gunning_fog_index,
    smog_index
)
from utils.formatting import color_code_index
from utils.tilmash_translation import tilmash_translate

def handle_translation():
    st.header("Перевод (Kazakh, Russian, English)")
    translate_input_method = st.radio("Способ ввода текста:", ["Загрузить файл", "Вставить текст"])
    input_text = ""

    if translate_input_method == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите файл (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
            input_text = read_file(temp_file_path)
            os.remove(temp_file_path)
            st.write("**Содержимое файла:**")
            st.write(input_text)
    else:
        input_text = st.text_area("Вставьте ваш текст здесь", height=200)

    if input_text:
        auto_detect = st.checkbox("Автоматически определить язык", value=True)
        src_lang = None
        if auto_detect:
            detected_lang = detect_language(input_text)
            if detected_lang in ['ru','en','kk']:
                st.info(f"Определён язык: {detected_lang}")
                src_lang = detected_lang
            else:
                st.warning("Не удалось определить язык. Выберите вручную.")
                src_lang = st.selectbox("Язык текста", ["ru", "en", "kk"])
        else:
            src_lang = st.selectbox("Язык текста", ["ru", "en", "kk"])

        if src_lang == "ru":
            tgt_options = ["en","kk"]
        elif src_lang == "en":
            tgt_options = ["ru","kk"]
        else:
            tgt_options = ["ru","en"]

        tgt_lang = st.selectbox("Перевод на:", tgt_options)

        if st.button("Перевести"):
            with st.spinner("Выполняется перевод..."):
                translated_text = tilmash_translate(input_text, src_lang, tgt_lang)
            if translated_text:
                st.subheader("Результат перевода:")
                st.write(translated_text)

                doc = Document()
                doc.add_paragraph(translated_text)
                doc_io = io.BytesIO()
                doc.save(doc_io)
                doc_io.seek(0)

                st.download_button(
                    label="Скачать переведённый текст (.docx)",
                    data=doc_io,
                    file_name="translated_text.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.warning("Не удалось выполнить перевод.")

def handle_readability_analysis():
    st.header("Анализ удобочитаемости текста")
    input_method = st.radio("Способ ввода текста:", ["Загрузить файл", "Вставить текст"])
    text = ""

    if input_method == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите файл (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
            text = read_file(temp_file_path)
            os.remove(temp_file_path)
            st.write("**Содержимое файла:**")
            st.write(text)
    else:
        text = st.text_area("Вставьте ваш текст здесь", height=200)

    if text:
        auto_detect = st.checkbox("Определить язык автоматически", value=True)
        if auto_detect:
            detected_lang = detect_language(text)
            st.info(f"Определён язык: {detected_lang}")
            lang_code = detected_lang if detected_lang in ['ru','en','kk'] else 'en'
        else:
            lang_code = st.selectbox("Язык текста", ["ru", "en", "kk"])

        if st.button("Анализировать"):
            fre = flesch_reading_ease(text, lang_code)
            fkgl = flesch_kincaid_grade_level(text, lang_code)
            fog = gunning_fog_index(text, lang_code)
            smog = smog_index(text, lang_code)

            st.subheader("Результаты удобочитаемости")
            st.markdown(
                f"**Индекс удобочитаемости Флеша:** {color_code_index('Flesch Reading Ease', fre)}",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Индекс Флеша-Кинкейда:** {color_code_index('Flesch-Kincaid Grade Level', fkgl)}",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Индекс тумана Ганнинга:** {color_code_index('Gunning Fog Index', fog)}",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Индекс SMOG:** {color_code_index('SMOG Index', smog)}",
                unsafe_allow_html=True
            )

def main():
    setup_nltk()

    st.title("Tilmash Translator & Readability Analysis")
    st.sidebar.header("Функциональность")
    functionality = st.sidebar.radio("Выберите режим:", ["Перевод", "Анализ удобочитаемости"])

    if functionality == "Перевод":
        handle_translation()
    elif functionality == "Анализ удобочитаемости":
        handle_readability_analysis()

if __name__ == "__main__":
    main()