import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import av
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
def load_huggingface_video(sample_rate: int = 60) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
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

    logger.info("🚀 Starting video decoding and subtitle parsing...")

    for entry in dataset:
        logger.info("📥 Entry received. Preparing to decode video and subtitles...")
        video_bytes = entry["mp4"]
        vtt_text = entry["en.vtt"].decode("utf-8") if isinstance(entry["en.vtt"], bytes) else entry["en.vtt"]
        video_stream = BytesIO(video_bytes)

        container = av.open(video_stream)
        stream = container.streams.video[0]
        time_base = stream.time_base
        logger.info("🎞️ Stream time base: %s | Sampling every %d frames", time_base, sample_rate)

        frame_count = 0
        used_frames = 0

        for i, frame in enumerate(tqdm(container.decode(video=0), desc="Decoding frames")):
            frame_count += 1
            if i % sample_rate != 0:
                continue

            img = frame.to_ndarray(format="rgb24")

            # Get timestamp, fallback to manual calculation
            if frame.pts is not None and time_base is not None:
                timestamp = float(frame.pts * time_base)
            elif frame.time is not None:
                timestamp = float(frame.time)
            else:
                timestamp = i / stream.average_rate if stream.average_rate else 0.0

            frames.append(img)
            timestamps.append(timestamp)
            used_frames += 1

        logger.info("✅ Decoded %d sampled frames out of %d total", used_frames, frame_count)

        logger.info("📝 Parsing subtitles from VTT...")
        for caption in WebVTT.from_string(vtt_text):
            subtitles.append({"start": caption.start, "end": caption.end, "text": caption.text.strip()})

        logger.info("📚 Parsed %d subtitle entries", len(subtitles))
        break  # Only process one video

    logger.info("🎯 Step complete. Returning frames, timestamps, and subtitles.")
    return frames, timestamps, subtitles
