from zenml import pipeline

from steps.etl import load_huggingface_video, process_video_chunks, save_to_mongo  # Import the new step


@pipeline
def video_etl_pipeline():
    # Step 1: Load video, decode frames, and extract subtitles
    frames, timestamps, subtitles = load_huggingface_video()

    # Step 2: Clean and chunk data for RAG
    frames, timestamps, subtitles = process_video_chunks(frames=frames, timestamps=timestamps, subtitles=subtitles)

    # Step 3: Save processed data to Mongo
    save_to_mongo(frames=frames, timestamps=timestamps, subtitles=subtitles)
