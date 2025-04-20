from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


def convert_timestamp_to_seconds(ts: str) -> float:
    """Convert WebVTT timestamp to seconds."""
    h, m, s = ts.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def align_frames_with_subtitles(frames: List[np.ndarray], timestamps: List[float], subtitles: List[Dict]) -> List[Dict]:
    aligned = []
    for i, timestamp in enumerate(timestamps):
        matched_texts = [
            sub["text"]
            for sub in subtitles
            if convert_timestamp_to_seconds(sub["start"]) <= timestamp <= convert_timestamp_to_seconds(sub["end"])
        ]
        aligned.append({"frame": frames[i], "timestamp": timestamp, "text": " ".join(matched_texts)})
    return aligned


def remove_redundant_frames(aligned_data: List[Dict], threshold: float = 0.95) -> List[Dict]:
    deduped = []
    last_vector = None

    for entry in aligned_data:
        frame_vector = entry["frame"].astype(np.float32).flatten()
        frame_vector /= np.linalg.norm(frame_vector) + 1e-8  # normalize

        if last_vector is None:
            deduped.append(entry)
            last_vector = frame_vector
            continue

        sim = cosine_similarity([frame_vector], [last_vector])[0][0]
        if sim < threshold:
            deduped.append(entry)
            last_vector = frame_vector

    logger.info("🧹 Removed %d redundant frames.", len(aligned_data) - len(deduped))
    return deduped


def merge_into_chunks(
    aligned_data: List[Dict], max_chars: int = 200
) -> Tuple[List[np.ndarray], List[float], List[str]]:
    frames, timestamps, texts = [], [], []
    buffer_text, buffer_frames, buffer_timestamps = "", [], []

    for entry in aligned_data:
        if not entry["text"]:
            continue

        if len(buffer_text) + len(entry["text"]) > max_chars:
            texts.append(buffer_text.strip())
            frames.append(buffer_frames[-1])  # pick last frame as representative
            timestamps.append(buffer_timestamps[-1])

            buffer_text, buffer_frames, buffer_timestamps = "", [], []

        buffer_text += " " + entry["text"]
        buffer_frames.append(entry["frame"])
        buffer_timestamps.append(entry["timestamp"])

    if buffer_text:
        texts.append(buffer_text.strip())
        frames.append(buffer_frames[-1])
        timestamps.append(buffer_timestamps[-1])

    return frames, timestamps, texts


@step
def process_video_chunks(
    frames: List[np.ndarray], timestamps: List[float], subtitles: List[Dict]
) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Match frames to subtitles, remove redundant frames using cosine similarity,
    and merge into ~200 character chunks for RAG input.
    """
    logger.info("🔗 Aligning frames with subtitles...")
    aligned = align_frames_with_subtitles(frames, timestamps, subtitles)

    logger.info("🧠 Removing redundant frames using cosine similarity...")
    deduped = remove_redundant_frames(aligned, threshold=0.95)

    logger.info("🧩 Merging frames/subtitles into ~200 character chunks...")
    merged_frames, merged_timestamps, merged_texts = merge_into_chunks(deduped)

    merged_subtitles = [
        {"start": str(ts), "end": str(ts), "text": text}
        for ts, text in zip(merged_timestamps, merged_texts, strict=False)
    ]

    logger.info("✅ Generated %d RAG-ready chunks.", len(merged_frames))
    return merged_frames, merged_timestamps, merged_subtitles
