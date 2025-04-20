from zenml import pipeline

from steps.etl import load_huggingface_video


@pipeline
def video_etl_pipeline():
    frames, timestamps, subtitles = load_huggingface_video()
