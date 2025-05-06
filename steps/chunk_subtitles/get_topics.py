import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


# --- HELPER FUNCTIONS ---


def merge_strings(str1, str2):
    max_overlap = 0
    min_len = min(len(str1), len(str2))
    for i in range(1, min_len + 1):
        if str1[-i:] == str2[:i]:
            max_overlap = i
    return str1 + str2[max_overlap:]


def recursive_merge_subtitle_texts(subtitle_texts):
    if len(subtitle_texts) == 0:
        return ""
    if len(subtitle_texts) == 1:
        return subtitle_texts[0]
    merged = []
    i = 0
    while i < len(subtitle_texts):
        if i + 1 < len(subtitle_texts):
            merged_pair = merge_strings(subtitle_texts[i], subtitle_texts[i + 1])
            merged.append(merged_pair)
        else:
            merged.append(subtitle_texts[i])
        i += 2
    return recursive_merge_subtitle_texts(merged)


def load_and_merge_file(file_path):
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            texts = [entry["text"].replace("\n", " ") for entry in data]
            return file_path.name, recursive_merge_subtitle_texts(texts)
    except Exception as e:
        logger.warning("Failed to load %s: %s", file_path.name, e)
        return file_path.name, ""


def merge_all_videos_subtitles(subtitles_dir: Path):
    subtitle_files = [file for file in sorted(subtitles_dir.iterdir()) if file.suffix == ".json"]
    logger.info("Found %d subtitle files to merge.", len(subtitle_files))

    merged_results = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_and_merge_file, file): file for file in subtitle_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging subtitles"):
            filename, merged_text = future.result()
            merged_results[filename] = merged_text
            logger.debug("Merged subtitles from %s (length: %d)", filename, len(merged_text))

    logger.info("Finished merging all subtitle files.")
    return merged_results


def overlapping_chunks(text, chunk_size=150, overlap=0):
    chunks = []
    start = 0
    while start < len(text):
        if start > 0:
            start = text.rfind(" ", 0, start) + 1
            if start == -1:
                start = 0
        end = start + chunk_size
        if end < len(text):
            end = text.rfind(" ", start, end)
            if end == -1:
                end = start + chunk_size
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def clean_decoded_text(text):
    text = re.sub(r"\b(\w+)(\s+\1){2,}", r"\1", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"(?:\s*\.\.\.){2,}", "...", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decode_chunk(chunk_text, tokenizer, model):
    try:
        inputs = tokenizer.encode("punctuate: " + chunk_text, return_tensors="tf", truncation=True)
        result = model.generate(inputs)
        decoded = tokenizer.decode(result[0], skip_special_tokens=True)
        return clean_decoded_text(decoded)
    except Exception as e:
        logger.warning("Error decoding chunk: %s", e)
        return ""


# --- ZENML STEP ---


@step
def get_topics() -> None:
    logger.info("Starting topic extraction from subtitles...")

    script_dir = Path(__file__).parent.parent.resolve()

    # Load models
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Merge subtitle files
    subtitles_path = script_dir / "subtitles_json"
    merged_subtitles = merge_all_videos_subtitles(subtitles_path)

    # Chunk the merged subtitles
    all_chunks = []
    for filename, merged_text in merged_subtitles.items():
        chunks = overlapping_chunks(merged_text)
        logger.debug("%s: split into %d chunks.", filename, len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total chunks to decode: %d", len(all_chunks))

    # Generate embeddings
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)

    # Fit BERTopic model
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(all_chunks, embeddings)

    # Save topic model to current dir
    topic_model.save(str(script_dir / "topic_model"))

    logger.info("Saved all files to %s", script_dir)
