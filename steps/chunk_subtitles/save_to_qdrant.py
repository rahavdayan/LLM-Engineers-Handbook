import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from zenml import step
from zenml.logger import get_logger


@step
def save_to_qdrant():
    logger = get_logger(__name__)

    # Define the directory using pathlib
    project_root = Path(__file__).resolve().parent.parent
    folder_path = project_root / "subtitle_chunks_json"

    if not folder_path.exists():
        logger.error("Directory does not exist: %s", folder_path)
        return

    # Collect subtitle chunks
    docs = []
    metadata = []

    for file_path in folder_path.iterdir():
        if file_path.suffix != ".json":
            continue
        try:
            with file_path.open("r", encoding="utf-8") as file:
                subtitle_data = json.load(file)
                for chunk in subtitle_data:
                    text = chunk.get("text", "")
                    docs.append(text)
                    metadata.append(chunk)  # full chunk as metadata
        except Exception as e:
            logger.warning("Failed to read file %s: %s", file_path.name, e)

    if not docs:
        logger.warning("No subtitle text found.")
        return

    # Generate embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "subtitle_chunks"

    # Create the collection if it doesn't exist
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
        )
        logger.info("Created new collection: %s", collection_name)

    # Prepare points
    points = [PointStruct(id=i, vector=embeddings[i], payload=metadata[i]) for i in range(len(docs))]

    # Upload to Qdrant
    client.upsert(collection_name=collection_name, points=points)
    logger.info("Successfully inserted %d subtitle embeddings into Qdrant.", len(points))


save_to_qdrant()
