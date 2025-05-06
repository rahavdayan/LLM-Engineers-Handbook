from datetime import datetime, timedelta
from pathlib import Path

import ffmpeg
import ollama
import yt_dlp
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# The chunk retrieval function
def grab_relevant_chunks(
    question, top_k=1, collection_name="subtitle_chunks", qdrant_host="localhost", qdrant_port=6333
):
    # Load the same embedding model used during indexing
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode(question)

    # Connect to Qdrant
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Perform similarity search
    search_results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)

    # Extract metadata/payloads
    top_metadata = [hit.payload for hit in search_results]

    return top_metadata


# Prompt Generation and Response
def generate_response(question, text):
    llm_prompt = f"""
    Here is the user question: "{question}"
    Here is text that could help them: "{text}"
    Please do the following:

    1. Summarize the main idea or key point explained in the question in one sentence.
    2. Highlight any explanation, definition, or key information related to the user's question.
    3. If relevant, rephrase technical descriptions into simpler or more understandable terms.
    """

    response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": llm_prompt}])

    response_content = response["message"]["content"]
    return response_content


def get_video_segment(FROM, TO, video_idx):
    # Create folder for storing downloaded videos
    video_dir = Path("videos")
    video_dir.mkdir(exist_ok=True)

    video_filename = video_dir / f"video_{video_idx}.mp4"
    output_path = Path("demo.mp4")

    # Parse input times
    fmt = "%H:%M:%S.%f"
    start_time = datetime.strptime(FROM, fmt)
    end_time = datetime.strptime(TO, fmt)
    base_duration = (end_time - start_time).total_seconds()

    # Extend to 20 seconds if shorter
    min_duration = 20
    if base_duration < min_duration:
        base_duration = min_duration

    # Check and download video if not already downloaded
    if not video_filename.exists():
        video_urls = {
            1: "https://youtu.be/XDW23i4xes0",
            2: "https://youtu.be/BsnqUV4yIL4",
            3: "https://youtu.be/hkcxTPL0sEE",
            4: "https://youtu.be/5FD0ZH6mGMs",
            5: "https://youtu.be/BdWI6DtCp4U",
            6: "https://youtu.be/ANeUQss6aTY",
            7: "https://youtu.be/RP3p2aQIORM",
            8: "https://youtu.be/8w4WKxd7GEc",
        }

        youtube_url = video_urls.get(video_idx)
        if not youtube_url:
            raise ValueError("Invalid video index.")

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": str(video_filename),
            "merge_output_format": "mp4",
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

    # Convert start_time to seconds
    start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6
    clip_start_str = str(timedelta(seconds=start_seconds))

    # Remove old output if it exists
    if output_path.exists():
        output_path.unlink()

    # Extract segment
    ffmpeg.input(str(video_filename), ss=clip_start_str, t=base_duration).output(
        str(output_path), vcodec="copy", acodec="copy"
    ).overwrite_output().run(quiet=False)

    return str(output_path)
