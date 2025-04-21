import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from webvtt import WebVTT
from zenml import step
from zenml.logger import get_logger

# Set up logger
logger = get_logger(__name__)

# Load .env from project root
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


@step
def load_huggingface_video(max_frames: int = 100) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    dataset_name = "aegean-ai/ai-lectures-spring-24"

    logger.info("🔑 Attempting to load Hugging Face access token...")
    try:
        hf_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
    except KeyError:
        logger.error("❌ HUGGINGFACE_ACCESS_TOKEN not found in environment variables.")
        raise EnvironmentError("HUGGINGFACE_ACCESS_TOKEN not found.") from None

    logger.info("✅ Logging into Hugging Face Hub...")
    login(token=hf_token)

    logger.info("📦 Streaming dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    frames, timestamps, subtitles = [], [], []

    for entry in dataset:
        logger.info("📥 Entry received. Preparing to decode video and subtitles...")

        video_bytes = entry["mp4"]
        vtt_text = entry["en.vtt"].decode("utf-8") if isinstance(entry["en.vtt"], bytes) else entry["en.vtt"]
        video_stream = BytesIO(video_bytes)

        # Save video to a temporary file (OpenCV requires a path)
        temp_dir = Path(os.getenv("TEMP_DIR", "/tmp"))
        temp_path = temp_dir / "temp_video.mp4"

        with temp_path.open("wb") as f:
            f.write(video_stream.read())

        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            logger.error("❌ Could not open video file with OpenCV.")
            raise RuntimeError("Failed to open video with OpenCV.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps

        logger.info("🎞️ Video info: %d total frames, %.2f FPS, %.2f seconds", total_frames, fps, duration_sec)

        # Compute frame indices to sample uniformly
        frame_indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)

        logger.info("🚶 Sampling frames using OpenCV...")
        for frame_idx in tqdm(frame_indices, desc="Grabbing frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Could not read frame at index %d", frame_idx)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_idx / fps)

        cap.release()
        logger.info("✅ Sampled %d frames using OpenCV.", len(frames))

        logger.info("📝 Parsing subtitles from VTT...")
        for caption in WebVTT.from_string(vtt_text):
            subtitles.append({"start": caption.start, "end": caption.end, "text": caption.text.strip()})

        logger.info("📚 Parsed %d subtitle entries", len(subtitles))
        break  # Only process one video

    logger.info("🎯 Step complete. Returning frames, timestamps, and subtitles.")
    return frames, timestamps, subtitles
