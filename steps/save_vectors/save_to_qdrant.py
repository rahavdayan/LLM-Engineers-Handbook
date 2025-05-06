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
    subtitles_dir = project_root / "subtitle_chunks_json"

    if not subtitles_dir.exists():
        logger.error("Directory does not exist: %s", subtitles_dir)
        return

    # Get all .json files in the directory and sort them
    subtitle_files = sorted([file for file in subtitles_dir.iterdir() if file.suffix == ".json"])

    # Collect subtitle chunks
    docs = []
    metadata = []

    for idx, file_path in enumerate(subtitle_files):
        # Assign video index based on the loop index (idx + 1)
        video_index = idx + 1  # Video index is 1-based

        try:
            with file_path.open("r", encoding="utf-8") as file:
                subtitle_data = json.load(file)
                for chunk in subtitle_data:
                    text = chunk.get("text", "")
                    docs.append(text)

                    # Include video index in metadata
                    chunk["video"] = video_index
                    metadata.append(chunk)
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

    # Always delete and recreate the collection to ensure it's clean
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        client.delete_collection(collection_name=collection_name)
        logger.info("Deleted existing collection: %s", collection_name)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
    )
    logger.info("Created fresh collection: %s", collection_name)

    # Prepare points
    points = [PointStruct(id=i, vector=embeddings[i], payload=metadata[i]) for i in range(len(docs))]

    # Upload to Qdrant
    client.upsert(collection_name=collection_name, points=points)
    logger.info("Successfully inserted %d subtitle embeddings into Qdrant.", len(points))
