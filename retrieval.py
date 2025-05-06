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
    INPUT = "input.mp4"
    OUTPUT = "demo.mp4"
    Path.unlink(OUTPUT)
    video_urls = {
        1: "https://youtu.be/_ekbcOBkMJU",
        2: "https://youtu.be/wOG5vNOTBsk",
        3: "https://youtu.be/xekV8aRFGuo",
        4: "https://youtu.be/m8c9hNEmJyc",
        5: "https://youtu.be/AqZQf1hiNok",
        6: "https://youtu.be/s7jDDNPNajI",
        7: "https://youtu.be/4XFANU-qJYE",
        8: "https://youtu.be/_xDS9TJXrEA",
    }

    youtube_url = video_urls.get(video_idx)
    if not youtube_url:
        raise ValueError("Invalid video index")

    # Download video using yt-dlp
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": INPUT,
        "merge_output_format": "mp4",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Parse input times
    fmt = "%H:%M:%S.%f"
    start_time = datetime.strptime(FROM, fmt)
    end_time = datetime.strptime(TO, fmt)

    # Midpoint of selected segment
    mid_point = start_time + (end_time - start_time) / 2
    ideal_clip_start = mid_point - timedelta(seconds=10)

    # Get full video duration using ffmpeg.probe
    probe = ffmpeg.probe(INPUT)
    video_duration = float(probe["format"]["duration"])  # in seconds

    # Clamp start and end to video bounds
    clip_start_sec = max(0, ideal_clip_start.timestamp())
    clip_end_sec = min(video_duration, clip_start_sec + 20)

    # Adjust start if clip is too close to end
    if clip_end_sec - clip_start_sec < 20:
        clip_start_sec = max(0, clip_end_sec - 20)

    # Format for ffmpeg input
    clip_start_str = str(timedelta(seconds=clip_start_sec))
    duration_str = str(timedelta(seconds=clip_end_sec - clip_start_sec))

    # Trim the video
    ffmpeg.input(INPUT, ss=clip_start_str, t=duration_str).output(
        OUTPUT, vcodec="copy", acodec="copy"
    ).overwrite_output().run()

    Path.unlink(INPUT)
    return OUTPUT
