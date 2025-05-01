from zenml import pipeline

from steps.chunk_subtitles import chunk  # Import the new step


@pipeline
def chunk_subtitles():
    chunk()
