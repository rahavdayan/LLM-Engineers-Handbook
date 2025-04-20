import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from torchvision.io import read_video
from zenml import step

# Load .env from project root
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def load_huggingface_video() -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Load video dataset from Hugging Face, decode frames, and extract subtitles.
    """
    dataset_name = "aegean-ai/ai-lectures-spring-24"

    # Get Hugging Face access token from environment variables
    try:
        hf_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
    except KeyError:
        raise EnvironmentError("HUGGINGFACE_ACCESS_TOKEN not found in environment variables.") from None

    login(token=hf_token)

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    frames, timestamps, subtitles = [], [], []

    for entry in dataset:
        video_bytes = entry["mp4"]
        video_stream = BytesIO(video_bytes)
        video, _, info = read_video(video_stream)

        for i in range(video.shape[0]):
            frame_data = video[i].numpy()
            timestamp = i / info["video_fps"]
            frames.append(frame_data)
            timestamps.append(timestamp)

        for subtitle in entry["en.vtt"]:
            subtitles.append({"start": subtitle["start"], "end": subtitle["end"], "text": subtitle["text"].strip()})

        logger.info("Processed video with %d frames and %d subtitles.", len(frames), len(subtitles))

    return frames, timestamps, subtitles
