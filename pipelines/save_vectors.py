from zenml import pipeline

from steps.save_vectors import save_to_qdrant


@pipeline
def save_vectors():
    save_to_qdrant()
