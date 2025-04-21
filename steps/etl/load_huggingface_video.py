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
def load_huggingface_video(sample_rate: int = 3) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
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
        duration = float(stream.duration * time_base) if stream.duration else 0.0
        fps = float(stream.average_rate)

        logger.info("🎞️ Stream info - Time base: %s | Duration: %.2f sec | FPS: %.2f", time_base, duration, fps)

        if duration == 0.0:
            logger.warning("⚠️ No duration found. Falling back to full decode...")
            break

        frame_interval_sec = sample_rate / fps
        timestamps_to_seek = np.arange(0, duration, frame_interval_sec)

        for seek_time in tqdm(timestamps_to_seek, desc="Smart seeking frames"):
            seek_pts = int(seek_time / time_base)
            container.seek(seek_pts, any_frame=False, backward=True, stream=stream)

            for frame in container.decode(video=0):
                frame_time = float(frame.pts * time_base) if frame.pts is not None else 0.0
                if abs(frame_time - seek_time) < (1.0 / fps):
                    img = frame.to_ndarray(format="rgb24")
                    frames.append(img)
                    timestamps.append(frame_time)
                    break  # move to next desired timestamp

        logger.info("✅ Sampled %d frames using smart seeking", len(frames))

        logger.info("📝 Parsing subtitles from VTT...")
        for caption in WebVTT.from_string(vtt_text):
            subtitles.append({"start": caption.start, "end": caption.end, "text": caption.text.strip()})

        logger.info("📚 Parsed %d subtitle entries", len(subtitles))
        break  # Only process one video

    logger.info("🎯 Step complete. Returning frames, timestamps, and subtitles.")
    return frames, timestamps, subtitles
