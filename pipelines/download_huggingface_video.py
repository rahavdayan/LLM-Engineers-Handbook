from zenml import pipeline

from steps.download_huggingface_video import save_video  # Import the new step


@pipeline
def download_huggingface_video():
    save_video()
