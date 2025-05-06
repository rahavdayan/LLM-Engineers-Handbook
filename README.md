# RAG System – CS370 Final Project

**by Rahav Dayan**

This project implements a RAG system with a user interface. The system is designed to take a user question (ex. “Explain what max pooling does”) and return relevant video segments with brief explanations for each. The application runs locally through a Gradio interface, which can be launched by running the `gradio.ipynb` notebook in the root directory.

## End-to-End System Overview

The system processes lecture videos from Hugging Face, aligns them with subtitles, applies topic modeling, and stores structured chunks in a vector database (Qdrant) for retrieval. A local language model (via Ollama) is used to generate user-friendly responses based on the retrieved content.

## Components

**`save_video.py`**  
This script downloads lecture videos from the Hugging Face dataset and extracts subtitle data. It stores each subtitle entry with its corresponding start and end timestamps for later alignment.

**`get_topics.py`**  
All subtitle files are merged into a single corpus and split into equal-length chunks. These are encoded using a sentence-transformer model and passed to BERTopic for topic modeling. The trained topic model is saved for downstream use.

**`chunk.py`**  
Subtitle text is cleaned and merged into larger blocks, removing redundant overlaps. A T5-based model is used to repunctuate the blocks, enabling accurate sentence segmentation via spaCy. Sentences are grouped into fixed-length chunks based on topic assignments from the saved model, along with their corresponding timestamps.

**`save_to_qdrant.py`**  
The sentence-level chunks are encoded into embeddings and stored in a Qdrant vector database for fast similarity-based retrieval.

**`gradio.ipynb`**  
This notebook defines the user interface and retrieval/generation logic. A user query is encoded, and the top-k relevant chunks are retrieved from Qdrant. These are passed to a locally hosted LLM (via Ollama) to generate a concise response along with related video segments.