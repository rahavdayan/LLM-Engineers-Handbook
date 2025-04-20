import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import av
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from webvtt import WebVTT
from zenml import step

# Load .env from project root
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


@step
def load_huggingface_video() -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Load one video sample from Hugging Face, decode the first frame, and extract subtitles.
    """
    dataset_name = "aegean-ai/ai-lectures-spring-24"

    try:
        hf_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
    except KeyError:
        raise EnvironmentError("HUGGINGFACE_ACCESS_TOKEN not found in environment variables.") from None

    login(token=hf_token)
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    frames, timestamps, subtitles = [], [], []

    for entry in dataset:
        video_bytes = entry["mp4"]
        vtt_text = entry["en.vtt"].decode("utf-8") if isinstance(entry["en.vtt"], bytes) else entry["en.vtt"]
        video_stream = BytesIO(video_bytes)

        # Decode first frame using PyAV
        container = av.open(video_stream)
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
            timestamps.append(frame.time)
            break  # Only one frame

        # Parse subtitles from VTT string (in memory)
        for caption in WebVTT.from_string(vtt_text):
            subtitles.append({"start": caption.start, "end": caption.end, "text": caption.text.strip()})

        break  # Only process one video

    return frames, timestamps, subtitles
