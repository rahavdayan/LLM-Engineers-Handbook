from zenml import pipeline

from steps.chunk_subtitles import chunk, get_topics


@pipeline
def chunk_subtitles():
    get_topics()
    chunk()
