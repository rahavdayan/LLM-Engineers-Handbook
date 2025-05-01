import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from webvtt import WebVTT
from zenml import step
from zenml.logger import get_logger


# Convert to timedelta
def convert_to_timedelta(timestamp):
    hours, minutes, rest = timestamp.split(":")
    if "." in rest:
        seconds, milliseconds = rest.split(".")
    else:
        seconds, milliseconds = rest, "0"
    return timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))


# Normalize timedelta to consistent string format
def timedelta_to_str(td):
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return "%02d:%02d:%09.6f" % (hours, minutes, seconds)


@step
def save_video():
    logger = get_logger(__name__)
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

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

    output_dir = project_root / "subtitles_json"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("🗑️ Existing subtitle directory removed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("📂 Subtitle directory created at %s", output_dir)

    idx = 1
    for entry in dataset:
        vtt_text = entry["en.vtt"].decode("utf-8") if isinstance(entry["en.vtt"], bytes) else entry["en.vtt"]

        logger.info("📝 Parsing subtitles from VTT...")
        subtitles = []
        for caption in WebVTT.from_string(vtt_text):
            start_td = convert_to_timedelta(str(caption.start))
            end_td = convert_to_timedelta(str(caption.end))

            # Skip subtitles shorter than or equal to 1 ms
            if (end_td - start_td) <= timedelta(milliseconds=10):
                continue

            # Normalize timestamps
            norm_start = timedelta_to_str(start_td)
            norm_end = timedelta_to_str(end_td)

            subtitles.append({"start": norm_start, "end": norm_end, "text": caption.text.strip()})

        logger.info("📚 Parsed %d subtitle entries", len(subtitles))

        json_path = output_dir / "subtitles_video_%d.json" % idx
        with Path.open(json_path, "w", encoding="utf-8") as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=2)
        logger.info("💾 Saved subtitles to %s", json_path)

        idx += 1

    logger.info("🎯 Step complete. Subtitles saved for %d videos.", idx - 1)


save_video()
