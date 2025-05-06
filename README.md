
# RAG System – CS370 Final Project

**by Rahav Dayan**

This project implements a RAG system with a user interface. The system is designed to take a user question and return relevant video segments with brief explanations for each. The application runs locally through a Gradio interface.

The system processes the lecture videos from Hugging Face, aligns them with subtitles, applies topic modeling, and stores structured chunks in a vector database (Qdrant) for retrieval. A local language model (via Ollama) is used to generate user-friendly responses based on the retrieved content.

It may take anywhere from a few seconds to a few minutes to receive a response.

## Quickstart

This repository is a fork of the textbook's repository, `LLM-Engineers-Handbook`. Here's how to get started once you clone the repo:

### 1. Install dependencies using Poetry
```bash
poetry install
```
If any dependencies are missing, you can manually add them:
```bash
poetry add <package_name>
```
Due to dependency conflicts when adding Gradio and Ollama via Poetry, these packages have been installed manually using pip instead. You can install them using:

```bash
pip install gradio
pip install ollama
```

### 2. Install Ollama and pull the Llama 3.1 model
You need to [install Ollama](https://ollama.com/download) for your operating system. Then run:
```bash
ollama pull llama3.1:8b
```

### 3. Start the ZenML server
```bash
poe local-infrastructure-up
```

### 4. Populate the Qdrant database
```bash
poe populate-qdrant
```

### 5. Run the Gradio app
Open and run the `gradio.ipynb` notebook in the root directory. You can either:

- Use the embedded Gradio interface within the notebook, or
- Visit [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your browser

## Components

**`save_video.py`**  
This script downloads lecture videos from the Hugging Face dataset and extracts subtitle data. It stores each subtitle entry with its corresponding start and end timestamps for later alignment.

**`get_topics.py`**  
All subtitle files are merged into a single corpus and split into equal-length blocks (150 characters each). These are encoded using a sentence-transformer model and passed to BERTopic for topic modeling. The trained topic model is saved for downstream use.

**`chunk.py`**  
Subtitle text is cleaned and merged into larger blocks (500 characters each), removing redundant overlaps. A T5-based model is used to repunctuate the blocks, enabling accurate sentence segmentation via spaCy. Sentences are grouped into approximately fixed-length chunks (~500 characters each) based on topic assignments from the saved model, along with their corresponding timestamps.

**`save_to_qdrant.py`**  
The sentence-level chunks are encoded into embeddings and stored in a Qdrant vector database for fast similarity-based retrieval.

**`retrieval.py`**  
This defines the retrieval/generation logic. A user query is encoded, and the top relevant chunk is retrieved from Qdrant. These are passed to a locally hosted LLM (via Ollama) to generate a concise response along with related video segments.

**`gradio.ipynb`**  
This notebook defines the user interface for the Gradio app. 

**`demonstration.ipynb`**  
This notebook highlights clips from the three sample questions mentioned in the assignment.

## Good Questions to Ask the RAG

Here are some additional questions I asked to assess my RAG:

- "Where is the yellow cab in the video?"
