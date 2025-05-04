from zenml import pipeline

from steps.chunk_subtitles import chunk, get_topics, save_to_qdrant


@pipeline
def chunk_subtitles():
    get_topics()
    chunk()
    save_to_qdrant()
