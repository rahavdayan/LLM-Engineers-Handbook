from typing import List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def grab_relevant_chunks(
    question: str,
    top_k: int = 5,
    collection_name: str = "subtitle_chunks",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> List[dict]:
    """
    Retrieve top-k most similar subtitle chunks from Qdrant for a given question.

    Args:
        question (str): The user query.
        top_k (int): Number of nearest neighbors to return.
        collection_name (str): Name of the Qdrant collection.
        qdrant_host (str): Qdrant host (default: localhost).
        qdrant_port (int): Qdrant port (default: 6333).

    Returns:
        List[dict]: List of metadata (payloads) for the most relevant vectors.
    """

    # Load the same embedding model used during indexing
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode(question)

    # Connect to Qdrant
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Perform similarity search
    search_results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)

    # Extract metadata/payloads
    top_metadata = [hit.payload for hit in search_results]

    return top_metadata
