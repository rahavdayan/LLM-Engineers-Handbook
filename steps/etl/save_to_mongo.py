import os
from typing import Dict, List

import bson
import gridfs
import numpy as np
from pymongo import MongoClient
from zenml import step
from zenml.logger import get_logger

# Set up logger
logger = get_logger(__name__)


@step
def save_to_mongo(frames: List[np.ndarray], timestamps: List[float], subtitles: List[Dict]) -> Dict[str, str]:
    """
    Save aligned video frame chunks (with timestamps and merged subtitle text)
    into a unified MongoDB collection using GridFS for frame storage.

    Args:
        frames: List of processed video frames (after de-duplication).
        timestamps: List of float timestamps associated with the frames.
        subtitles: List of dictionaries with 'start', 'end', and 'text' for each chunk.

    Returns:
        Dictionary with MongoDB connection details.
    """
    mongo_uri = os.getenv("DATABASE_HOST", "mongodb://localhost:27017")
    db_name = "video_data"
    unified_collection_name = "rag_chunks"

    logger.info("🔌 Connecting to MongoDB...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)

    collection = db[unified_collection_name]
    logger.info("🧹 Clearing previous collection (if any)...")
    collection.drop()

    documents = []
    logger.info("💾 Storing unified video RAG chunks...")

    for i, (frame, timestamp, subtitle) in enumerate(zip(frames, timestamps, subtitles, strict=False)):
        # Store frame in GridFS
        frame_binary = bson.binary.Binary(frame.tobytes())
        file_id = fs.put(frame_binary, filename="chunk_frame_" + str(i + 1) + ".png")

        documents.append(
            {
                "chunk_number": i + 1,
                "frame_id": file_id,
                "timestamp": timestamp,
                "subtitle": subtitle["text"],
                "start_time": subtitle["start"],
                "end_time": subtitle["end"],
            }
        )

    collection.insert_many(documents)

    logger.info("✅ Successfully saved %d RAG chunks to MongoDB.", len(documents))
    return {"database": db_name, "unified_collection": unified_collection_name}
