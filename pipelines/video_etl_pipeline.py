from zenml import pipeline

from steps.etl import load_huggingface_video


@pipeline
def video_etl_pipeline():
    # Step 1: Load raw bytes from Hugging Face dataset (video + subtitles)
    mp4_bytes, vtt_bytes = load_huggingface_video()
