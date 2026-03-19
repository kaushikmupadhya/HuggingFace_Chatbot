# EduPoint – Practitioner & Patient Chat Help Bot

A conversational AI assistant built with Streamlit and open-source LLMs, designed to support healthcare practitioners and patients with contextual, chat-based help.

---

## Overview

This app provides a simple chat interface powered by a locally-run or HuggingFace-hosted language model. It maintains conversation history across turns, making interactions feel natural and context-aware.

The project includes two LLM backend options:

- **Falcon 7B Instruct** (active) — loaded via HuggingFace Transformers, runs on CPU or GPU
- **LLaMA 2 7B Chat GGUF** (commented out) — lightweight quantized model via `llama-cpp-python` and LlamaIndex

---

## Project Structure
```
.
├── app.py       # Main Streamlit app with ChatBot class and UI logic
├── main.py      # Script to download the LLaMA GGUF model weights
└── README.md
```

## Getting Started

### 1. Install dependencies
```bash
pip install streamlit transformers torch
```

For the LLaMA backend (optional):
```bash
pip install llama-index-llms-llama-cpp llama-index-embeddings-huggingface
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. (Optional) Download LLaMA model weights
```bash
python main.py
```

This downloads `llama-2-7b-chat.Q2_K.gguf` from HuggingFace into the current directory.

---

## Configuration

| Setting | Default | Notes |
|---|---|---|
| Model | `tiiuae/falcon-7b-instruct` | Swap in `ChatBot.__init__()` |
| Max tokens | `2000` | Adjust in `generate_response()` |
| Temperature | `0.4` | Controls response randomness |
| Device | Auto-detected | Uses CUDA if available, else CPU |

---

## Notes

- GPU is strongly recommended for Falcon 7B — CPU inference will be slow
- Chat history is stored in Streamlit session state and can be cleared via the sidebar
- The LLaMA-based backend (in `app.py` comments) requires the GGUF model to be present locally before running

---

## Intended Use

Built for educational and informational support in healthcare contexts. Not a substitute for professional medical advice.
