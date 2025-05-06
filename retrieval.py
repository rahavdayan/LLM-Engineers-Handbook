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

    # Check and download video if not already downloaded
    if not video_filename.exists():
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
            raise ValueError("Invalid video index.")

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": str(video_filename),
            "merge_output_format": "mp4",
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

    # Get full video duration
    probe = ffmpeg.probe(str(video_filename))
    video_duration = float(probe["format"]["duration"])

    # Parse input times
    fmt = "%H:%M:%S.%f"
    start_time = datetime.strptime(FROM, fmt)
    end_time = datetime.strptime(TO, fmt)
    base_duration = (end_time - start_time).total_seconds()

    # Ensure at least 20 seconds but not beyond video end
    desired_duration = max(20, base_duration)

    # Convert start_time to seconds
    start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6

    # Cap duration to video length
    max_possible_duration = video_duration - start_seconds
    final_duration = min(desired_duration, max_possible_duration)

    if final_duration <= 0:
        raise ValueError("Start time is beyond the end of the video.")

    clip_start_str = str(timedelta(seconds=start_seconds))

    # Remove old output if it exists
    if output_path.exists():
        output_path.unlink()

    # Extract segment
    ffmpeg.input(str(video_filename), ss=clip_start_str, t=final_duration).output(
        str(output_path), vcodec="copy", acodec="copy"
    ).overwrite_output().run()

    return str(output_path)
