
# RAG System – CS370 Final Project

**by Rahav Dayan**

This project implements a RAG system with a user interface. The system is designed to take a user question and a relevant video segment with a brief explanation of it. The application runs locally through a Gradio interface.

The system downloads the lecture videos from Hugging Face, applies topic modeling and chunks subtitle texts, and stores the structured chunks in a vector database (Qdrant) for retrieval. A local language model (via Ollama) is used to generate user-friendly responses based on the retrieved content.

It may take anywhere from a few seconds to a few minutes to receive a response.

## Quickstart

This repository is a fork of the original ```LLM-Engineers-Handbook``` textbook repository. After cloning the repo, make sure you have all the required dependencies from the original project installed. In addition, follow the steps below to get started with the specific additions for this project.

### 1. Set up Environment Variables

Change the name of the `.env.example` file to `.env` to set up your environment variables.

### 2. Install dependencies using Poetry
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

### 3. Install Ollama and pull the Llama 3.1 model
You need to [install Ollama](https://ollama.com/download) for your operating system.

Ollama may start automatically in the background (especially on macOS if installed via the desktop app). Before starting it manually, check if it's already running by visiting:

[http://127.0.0.1:11434](http://127.0.0.1:11434)

If the server is running, you'll see a JSON response like:

```json
{ "status": "ok" }
```

If you don't see that response or the page fails to load, start the Ollama server manually by running:

```bash
ollama serve
```

After you get the server running, run the following:
```bash
ollama pull llama3.1:8b
```

### 4. Start the ZenML server
```bash
poe local-infrastructure-up
```

### 5. Populate the Qdrant database
```bash
poe populate-qdrant
```

### 6. Run the Gradio app
Open and run the `gradio.ipynb` notebook in the root directory. You can either:

- Use the embedded Gradio interface within the notebook, or
- Visit [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your browser

## Components

**`save_video.py`**  
This script downloads lecture videos from the Hugging Face dataset and extracts subtitle data. It stores each subtitle text with its corresponding start and end timestamps for later alignment.

**`get_topics.py`**  
All subtitle files are merged into a single corpus and split into equal-length blocks (150 characters each). These are encoded using a sentence-transformer model and passed to BERTopic for topic modeling. The trained topic model is saved for downstream use.

**`chunk.py`**  
Subtitle text is cleaned and merged into larger blocks (500 characters each), removing redundant overlaps. A T5-based model is used to repunctuate the blocks, enabling accurate sentence segmentation via spaCy. Sentences are grouped into approximately fixed-length chunks (~500 characters each) based on topic assignments from the saved model, along with their corresponding timestamps.

**`save_to_qdrant.py`**  
The sentence-level chunks are encoded into embeddings and stored in a Qdrant vector database for fast similarity-based retrieval.

**`retrieval.py`**  
This defines the retrieval/generation logic. A user query is encoded, and the top relevant chunk is retrieved from Qdrant. It is passed to a locally hosted LLM (via Ollama) to generate a concise response along with the associated video segment.

**`gradio.ipynb`**  
This notebook defines the user interface for the Gradio app. 

**`demonstration.ipynb`**  
This notebook highlights the retrieved clips from the three sample questions mentioned in the assignment.

## Good Questions to Ask the RAG

Here are some additional questions I asked to assess my RAG:

- "Where is the yellow cab in the video?"
