# Tilmash Translator

**Tilmash Translator** is an offline‑first, privacy‑preserving translation and readability toolkit for Russian, English and Kazakh.

It ships as a Streamlit web‑app and offers two core capabilities:

1. **Neural Machine Translation**  
   • Primary model — [ISSAI/tilmash](https://huggingface.co/issai/tilmash) (Seq2Seq) for RU ↔ EN ↔ KK  
   • Long‑text fallback — *Gemma‑3* 12B (GGUF) running locally with `llama‑cpp-python` (+ optional GPU layers)  
   • Smart chunking & streaming make multi‑page documents feel snappy
2. **Readability Analysis**  
   • Calculates Flesch Reading Ease, Flesch‑Kincaid, Gunning Fog and SMOG  
   • Highlights complex words and supports RU/EN/KK


---

## Quick Start

```bash
# 1. Clone & create a virtual environment
$ git clone https://github.com/medetshatayev/Tilmash_Translator.git
$ cd Tilmash_Translator
$ python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. (optional) authenticate once to download the Tilmash weights
$ echo "HF_TOKEN=🪄your_huggingface_token" > .env

# 4. Launch the Streamlit app
$ streamlit run main.py
```

💡 The helper script `start.sh` automates the above and sets safe memory limits for `llama‑cpp-python`.

### GPU Off‑loading (Gemma‑3)

Set `GEMMA_GPU_LAYERS=<num_layers>` in your environment (defaults to **48**) to off‑load those layers to Metal/CUDA.

---

## Project Layout

```
.
├── main.py               # Streamlit UI
├── utils/                # Translation & analysis helpers
│   ├── tilmash_translation.py
│   ├── gemma_translation.py
│   ├── readability_indices.py
│   └── ...
├── models/               # Extra resources (NLTK, etc.)
├── config.py             # Default env‑vars
├── start.sh              # Convenience launcher
└── requirements.txt      # Python deps
```

## Configuration Keys

| Variable               | Default | Purpose                                   |
|------------------------|---------|-------------------------------------------|
| `GEMMA_GPU_LAYERS`     | 48      | Layers to move to GPU (0 = CPU‑only)      |
| `GEMMA_CONTEXT_SIZE`   | 8192    | Context window for Gemma‑3                |
| `MAX_PARALLEL_MODELS`  | 4       | Concurrency guard                         |
| `MAX_TOKENS`           | 4096    | Generation cap per request                |
| `CHUNK_SIZE`           | 3000    | Token threshold before auto‑chunking      |

Override any of these via the environment or edit **config.py**.

---

## How It Works

1. **File ingestion** — `.txt`, `.docx`, `.pdf` loaded via `utils/file_readers.py`  
2. **Language detection** — `langdetect` (auto‑detect option in UI)  
3. **Translation pipeline** — <3000 tokens translate directly; longer texts are chunked (`utils/chunking.py`) and streamed through Tilmash or Gemma‑3  
4. **Readability analysis** — scores computed in `utils/readability_indices.py` and color‑coded in the app.

---

## License

Distributed under the MIT License — see `LICENSE` for details.

## Acknowledgements

- ISSAI team for releasing **Tilmash**  
- Meta for open‑sourcing *Gemma‑3*  
- Streamlit, Hugging Face and the Python OSS community