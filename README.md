# RAG-Ingest: PDF to Markdown Extraction and Indexing for RAG

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [How It Works](#how-it-works)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)

## Introduction

RAG-Ingest is a powerful tool designed to extract markdown content from PDF files and index it in a vector database (Qdrant) for Retrieval Augmented Generation (RAG). This project aims to streamline the process of converting PDF documents into a format suitable for advanced natural language processing tasks.

## Features

- PDF to Markdown conversion with layout preservation
- Image extraction and captioning
- Table detection and conversion to markdown format
- Intelligent header detection
- Code block identification and language detection
- Vector indexing using Qdrant for efficient retrieval
- Support for multiple PDF processing
- Configurable settings for fine-tuning the extraction process

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended for faster processing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/alt-gan/rag-ingest.git
   cd rag-ingest
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Configuration

1. Create a `.env` file in the project root and add the following environment variables:
   ```
   TG_API_KEY=your_together_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

2. Modify the `config/config.yaml` file to adjust extraction and indexing settings:

3. Customize the prompts in `config/prompts.json` for context-aware chunk modification.

## Usage

To extract markdown from a PDF and index it:
```python 
index.py --input path/to/your/file.pdf --file_category category_name --collection_name rag_llm --persist_dir persist
````


Arguments:
- `--input`: Path to the input PDF file or directory
- `--file_category`: Category of the file (finance, healthcare, or oil_gas)
- `--collection_name`: Name of the Qdrant collection (default: rag_llm)
- `--persist_dir`: Directory to persist the index (default: persist)
- `--md_flag`: Flag to process markdown files instead of PDFs


## Project Structure

```bash
├── config/
│ ├── config.yaml
│ └── prompts.json
├── extract.py
├── index.py
├── requirements.txt
└── README.md
```

## How It Works
1. PDF Extraction (`extract.py`):
   - The `MarkdownPDFExtractor` class handles the conversion of PDF to markdown.
   - It uses PyMuPDF to extract text and layout information.
   - Images are extracted, saved, and captioned using a pre-trained model.
   - Tables are detected and converted to markdown format.
   - Code blocks are identified and language is detected.

2. Indexing (`index.py`):
   - The `Index` class manages the indexing process.
   - It uses LlamaIndex and Qdrant for vector storage and retrieval.
   - Documents are split into chunks and processed in parallel.
   - Anthropic's Claude model is used for context-aware chunk modification.
   - Processed chunks are indexed in the Qdrant vector store.

## Customization

- Adjust extraction parameters in `config/config.yaml`
- Modify prompts for chunk modification in `config/prompts.json`
- Extend the `PDFExtractor` class in `extract.py` for custom extraction logic
- Implement additional vector stores by modifying the `Index` class in `index.py`

## Troubleshooting

- If you encounter CUDA out of memory errors, try reducing the batch size or using a smaller model.
- For OCR issues, ensure Tesseract is correctly installed and its path is set in the system environment variables.
- If indexing is slow, consider using a more powerful GPU or increasing the number of worker threads.
