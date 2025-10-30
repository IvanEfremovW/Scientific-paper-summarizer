# Long document summarization with Chain of Density prompting

[English](README.md) | [Русский](README_RU.md)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/Langcahin-00a67e?logo=langchain)](https://langchain.com)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-white?logo=gradio)](https://www.gradio.app/)

## Overview

This project implemetns an end-to-end system that automatically generates concise and accurate abstracts from long-text documents.
It leverages the **Phi-3-mini-128k-instruct** language model and a **Map-Reduce + Chain-of-Density** pipeline to handle documents of any length while preserving factual integrity and relevance.

## Key Features

- **Accurate Parsing**
  Uses **PyMuPDF** to reliably extract text from complex layouts (multi-column, multi-page).

- **Scalable Summarization via Map-Reduce**  
    Handles documents of any length by splitting text into chunks, summarizing each, and intelligently combining results into a coherent abstract.

- **Enhanced Quality with Chain-of-Density (CoD) prompting**  
    Iteratively compresses the summary to maximize information density while removing redundancy and generic phrasing.
## Project structure

```Text
Long-document-CoD-summarizer/
├── src/
|   └── summarizer/
│       ├── app.py              # Gradio UI
│       ├── config.py           # Model configuration
│       ├── ingestion.py        # Document parsing
│       └── summarizer.py       # LLM model logic
├── tests/                  
├── Dockerfile
├── .env.example            # Example for .env file for config
├── pyproject.toml          
└── requirements.txt        
```

## 🚀 Getting started

### 1. Clone the Repository
```bash
https://github.com/IvanEfremovW/Long-document-CoD-summarizer.git
cd Long-document-CoD-summarizer
```

### 2.1 Run with Docker (Recommended) 

#### Build the Docker image
```bash
docker build -t Long-document-CoD-summarizer .
```
#### Run on GPU
```bash
docker run -it --rm --gpus all -p 7860:7860 Long-document-CoD-summarizer
```
#### Run on CPU (slower, for testing only)
```bash
docker run -it --rm -p 7860:7860 Long-document-CoD-summarizer
```

### 2.2 Alternatively run without Docker
>⚠️ Requires Python 3.10+, CUDA 12.1 and compatible version of PyTorch to run on GPU. 
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
python -m src.summarizer.app
```

### 3. Open the web interface
Once the container is running, open your browser and go to:
👉 http://127.0.0.1:7860

You’ll see the Gradio interface:

1. Click “Choose File”
2. Upload a document
3. Wait for the abstract to be generated
