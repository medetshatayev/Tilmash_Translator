#!/bin/bash

# Change directory to your project folder
cd /Users/mshatayev/PycharmProjects/Tilmash_Translator

# Activate the virtual environment
source .venv/bin/activate

# Start the Streamlit app
streamlit run main.py --server.port 8501 --server.fileWatcherType none --server.address 0.0.0.0