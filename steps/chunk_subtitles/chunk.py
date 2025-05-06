import json
import re
import shutil
import string
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from zenml import step
from zenml.logger import get_logger


def convert_to_timedelta(timestamp):
    """Converts a timestamp string (HH:MM:SS.ssssss) to timedelta."""
    time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")
    return timedelta(
        hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second, microseconds=time_obj.microsecond
    )


def timedelta_to_str(td):
    if td is not None:
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return "%02d:%02d:%09.6f" % (hours, minutes, seconds)
    return "%02d:%02d:%09.6f" % (0, 0, -1)


def get_non_overlapping_part(str1, str2):
    max_overlap = 0
    min_len = min(len(str1), len(str2))
    for i in range(1, min_len + 1):
        if str1[-i:] == str2[:i]:
            max_overlap = i
    return str2[max_overlap:].strip()


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


def group_subtitles_by_char_limit(subtitles, max_chars=500):
    """
    Groups subtitles into chunks based on a maximum character limit.
    Tracks original subtitle data for each chunk. Timestamps are now stored as timedeltas.
    """
    chunks = []
    current_chunk = {
        "text": "",  # Accumulated chunk text
        "start": timedelta.max,  # Start time of the chunk (earliest) as a timedelta
        "end": timedelta.min,  # End time of the chunk (latest) as a timedelta
        "subtitles": [],  # List to track original subtitles in this chunk
    }

    for subtitle in subtitles:
        subtitle_text = subtitle["text"]
        subtitle_start = subtitle["start"]  # Already in timedelta
        subtitle_end = subtitle["end"]  # Already in timedelta

        # Check if adding this subtitle exceeds the character limit
        if len(current_chunk["text"]) > max_chars:
            # Save current chunk and start a new one
            chunks.append(current_chunk)
            current_chunk = {
                "text": get_non_overlapping_part(current_chunk["text"], subtitle_text),
                "start": subtitle_start,
                "end": subtitle_end,
                "subtitles": [subtitle],
            }
        else:
            # Add to current chunk
            current_chunk["text"] = merge_strings(current_chunk["text"], subtitle_text)
            current_chunk["start"] = min(current_chunk["start"], subtitle_start)
            current_chunk["end"] = max(current_chunk["end"], subtitle_end)
            current_chunk["subtitles"].append(subtitle)

    # Add the last chunk
    if current_chunk["text"]:
        chunks.append(current_chunk)

    return chunks


def clean_subtitle_text(subtitle_text):
    filler_words = [r"\buh\b", r"\bum\b", r"\ber\b", r"\bah\b", r"\bhmm\b"]
    filler_pattern = re.compile(r"|".join(filler_words), re.IGNORECASE)
    text = subtitle_text.replace("\n", " ")
    text = filler_pattern.sub("", text)
    text = text.translate(str.maketrans("", "", string.punctuation)).strip().lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def combine_subtitles(filepath):
    with Path.open(filepath, "r") as file:
        data = json.load(file)

    # Convert the timestamps from string to timedelta
    for item in data:
        item["start"] = convert_to_timedelta(item["start"])  # Convert to timedelta
        item["end"] = convert_to_timedelta(item["end"])  # Convert to timedelta
        item["text"] = clean_subtitle_text(item["text"])

    # Group subtitles by character limit, now with correct timedelta timestamps
    combined_subtitles = group_subtitles_by_char_limit(data)
    return combined_subtitles


def clean_sentences(sent_lst):
    cleaned = []
    for sentence in sent_lst:
        # Remove punctuation using str.translate
        no_punct = sentence.translate(str.maketrans("", "", string.punctuation)).strip().lower()
        cleaned.append(no_punct)
    return cleaned


def clean_decoded_text(text):
    text = re.sub(r"\b(\w+)(\s+\1){2,}", r"\1", text)  # remove repeated words
    text = re.sub(r"\.{3,}", "...", text)  # limit long ellipses
    text = re.sub(r"(?:\s*\.\.\.){2,}", "...", text)  # collapse multiple ellipses
    text = re.sub(r"\s+", " ", text).strip()  # normalize spacing
    return text


def get_word_timings(subtitle, marker_length=10):
    word_timings = []
    for entry in subtitle:
        word_timings.append({"word": entry["text"][:marker_length], "timestamp": entry["start"]})
        word_timings.append({"word": entry["text"][-marker_length:], "timestamp": entry["end"]})
    return word_timings


def separate_into_chunks(
    combined_subtitles, sents_lst, topic_model, embedding_model, min_chars=400, max_chars=600, time_jump_threshold=10
):
    chunks = []
    current_chunk = []
    current_length = 0
    prev_topic = None

    # Get embeddings and topics for all sentences first
    embeddings = embedding_model.encode(sents_lst)
    topics, _ = topic_model.transform(sents_lst, embeddings)

    for i, sent_text in enumerate(sents_lst):
        current_topic = topics[i]

        # Decide whether to start a new chunk
        if (current_length + len(sent_text) + 1 > min_chars) and (
            (current_length + len(sent_text) + 1 > max_chars)
            or (prev_topic is not None and current_topic != prev_topic)
        ):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        # Add current sentence to chunk
        current_chunk.append(sent_text)
        current_length += len(sent_text) + 1
        prev_topic = current_topic

    # Add the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    current_subtitle_idx = 0
    word_timings = get_word_timings(combined_subtitles[current_subtitle_idx]["subtitles"])
    subtitle_chunks = []
    last_used_ts = timedelta(seconds=0)  # Initial starting point (0:00)

    for i in range(0, len(chunks)):
        last_seen_ts = None
        matching = [entry for entry in word_timings if entry["word"] in chunks[i] and entry["timestamp"] > last_used_ts]

        if len(matching) == 0:
            if i < len(chunks) - 1:  # lookahead to next chunk
                next_chunk = chunks[i + 1]
                matching = [
                    entry for entry in word_timings if entry["word"] in next_chunk and entry["timestamp"] > last_used_ts
                ]
                if i < len(chunks) - 2:  # lookahead to next next chunk
                    next_chunk = chunks[i + 2]
                    matching = [
                        entry
                        for entry in word_timings
                        if entry["word"] in next_chunk and entry["timestamp"] > last_used_ts
                    ]
                    if i < len(chunks) - 3:  # lookahead to next next next chunk
                        next_chunk = chunks[i + 3]
                        matching = [
                            entry
                            for entry in word_timings
                            if entry["word"] in next_chunk and entry["timestamp"] > last_used_ts
                        ]

            if len(matching) == 0:
                current_subtitle_idx += 1
                if current_subtitle_idx <= len(combined_subtitles) - 1:
                    word_timings = get_word_timings(combined_subtitles[current_subtitle_idx]["subtitles"])
                    matching = [
                        entry
                        for entry in word_timings
                        if entry["word"] in chunks[i] and entry["timestamp"] > last_used_ts
                    ]

        if len(matching) > 0:
            start_ts = min(min(entry["timestamp"] for entry in matching), last_used_ts)
            if len(subtitle_chunks) > 0 and subtitle_chunks[-1]["end"] is None:
                subtitle_chunks[-1]["end"] = start_ts
            end_ts = max(entry["timestamp"] for entry in word_timings)
            for match in matching:
                candidate_ts = match["timestamp"]

                if last_seen_ts is None:
                    last_seen_ts = candidate_ts
                    best_end_ts = candidate_ts
                elif candidate_ts - last_seen_ts <= timedelta(seconds=time_jump_threshold):
                    last_seen_ts = candidate_ts
                    best_end_ts = candidate_ts
                # else: skip this timestamp, don't update best_end_ts

            end_ts = best_end_ts
            last_used_ts = end_ts if end_ts is not None else last_used_ts
        else:
            start_ts = end_ts
            end_ts = None

        subtitle_chunks.append({"start": start_ts, "end": end_ts, "text": chunks[i]})
    if len(subtitle_chunks) > 0 and subtitle_chunks[-1]["end"] is None:
        subtitle_chunks[-1]["end"] = max(entry["timestamp"] for entry in word_timings)
    return subtitle_chunks


def process_subtitles_file(
    filepath,
    idx,
    tokenizer,
    model,
    nlp,
    topic_model,
    embedding_model,
    output_dir,
    logger,
    repetition_penalty=2.5,
    max_new_tokens=256,
):
    try:
        logger.info("Processing: %s", filepath)

        # Combine subtitles based on char limit logic
        combined_subtitles = combine_subtitles(filepath)

        # 1. Loop through each combined chunk and apply punctuation restoration
        decoded_chunks = []
        for chunk in combined_subtitles:
            chunk_text = chunk["text"]

            # Add punctuation using your model
            inputs = tokenizer.encode("punctuate: " + chunk_text, return_tensors="tf", truncation=True)
            result = model.generate(inputs, repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens)
            decoded = tokenizer.decode(result[0], skip_special_tokens=True)

            # Clean the decoded text (you can modify this based on your needs)
            cleaned = clean_decoded_text(decoded)

            # Append the cleaned, punctuated chunk to the list
            decoded_chunks.append(cleaned)

        # Merge all decoded chunks into a single text blob
        merged_chunks = recursive_merge_subtitle_texts(decoded_chunks)

        # 2. Use NLP model to clean up sentence structure
        doc = nlp(merged_chunks)

        # Clean and split sentences
        sent_lst = clean_sentences([sent.text for sent in doc.sents])

        # 3. Separate into final chunks based on sentences, topic modeling, and embeddings
        final_chunks = separate_into_chunks(combined_subtitles, sent_lst, topic_model, embedding_model)
        for chunk in final_chunks:
            chunk["start"] = timedelta_to_str(chunk["start"])
            chunk["end"] = timedelta_to_str(chunk["end"])

        # 4. Save the final chunks to a JSON file
        json_path = output_dir / f"subtitles_chunks_video_{idx + 1}.json"
        with Path.open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)

        logger.info("✅ Saved chunked subtitles to %s", json_path)

    except Exception as e:
        logger.error("❌ Failed to process %s: %s", filepath, str(e))


@step
def chunk():
    logger = get_logger(__name__)

    # Prepare output directory
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "subtitle_chunks_json"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("🗑️ Existing subtitle_chunks_json directory removed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("📂 Subtitle_chunks_json directory created at %s", output_dir)

    subtitles_dir = project_root / "subtitles_json"
    subtitle_files = sorted([file for file in subtitles_dir.iterdir() if file.suffix == ".json"])

    # Load models
    tokenizer = T5Tokenizer.from_pretrained("SJ-Ray/Re-Punctuate")
    model = TFT5ForConditionalGeneration.from_pretrained("SJ-Ray/Re-Punctuate")
    topic_model = BERTopic.load(project_root / "topic_model")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = spacy.load("en_core_web_sm")

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each subtitle file
        for idx, filepath in enumerate(subtitle_files):
            executor.submit(
                process_subtitles_file,
                filepath,
                idx,
                tokenizer,
                model,
                nlp,
                topic_model,
                embedding_model,
                output_dir,
                logger,
            )
