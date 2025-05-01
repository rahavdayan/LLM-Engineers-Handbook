import json
import os
import re
import shutil
import string
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import spacy
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from zenml import step
from zenml.logger import get_logger


def convert_to_float_seconds(timestamp):
    hours, minutes, rest = timestamp.split(":")
    if "." in rest:
        seconds, milliseconds = rest.split(".")
    else:
        seconds, milliseconds = rest, "0"

    total_seconds = (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / (10 ** len(milliseconds))  # Handles .1, .12, .123, etc.
    )
    return float(total_seconds)


def float_to_timestamp_str(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return "%02d:%02d:%09.6f" % (hours, minutes, secs)


def merge_strings(str1, str2):
    max_overlap = 0
    min_len = min(len(str1), len(str2))
    for i in range(1, min_len + 1):
        if str1[-i:] == str2[:i]:
            max_overlap = i
    return str1 + str2[max_overlap:]


def parse_ts(ts):
    return datetime.strptime(ts, "%H:%M:%S.%f") - datetime.strptime("0:00:00.000000", "%H:%M:%S.%f")


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


def recursive_merge_subtitles(subtitles, tolerance_ms=10):
    if not subtitles:
        return {}
    if len(subtitles) == 1:
        return subtitles[0]

    merged = []
    i = 0
    while i < len(subtitles):
        if i + 1 < len(subtitles):
            s1 = subtitles[i]
            s2 = subtitles[i + 1]

            # Merge text with overlap handling
            merged_text = merge_strings(s1["text"], s2["text"])

            # Extract all word/timestamp pairs
            timestamps = []
            for key in s1:
                if key.startswith("T") and isinstance(s1[key], dict):
                    timestamps.append((s1[key]["word"], s1[key]["timestamp"]))
            for key in s2:
                if key.startswith("T") and isinstance(s2[key], dict):
                    timestamps.append((s2[key]["word"], s2[key]["timestamp"]))

            # Deduplicate using time tolerance
            seen = []
            for word, ts in timestamps:
                current_time = parse_ts(ts)
                is_unique = True  # Assume the word is unique unless proven otherwise
                for entry in seen:
                    if abs(current_time - parse_ts(entry["timestamp"])) <= timedelta(milliseconds=tolerance_ms):
                        is_unique = False  # Found a duplicate, break the loop
                        break
                if is_unique:
                    seen.append({"word": word, "timestamp": ts})

            # Build merged subtitle object
            merged_sub = {"T%d" % j: entry for j, entry in enumerate(seen)}
            merged_sub["text"] = merged_text
            merged.append(merged_sub)
        else:
            merged.append(subtitles[i])
        i += 2
    return recursive_merge_subtitles(merged, tolerance_ms=tolerance_ms)


def clean_subtitle_text(subtitle_text):
    filler_words = [r"\buh\b", r"\bum\b", r"\ber\b", r"\bah\b", r"\bhmm\b"]
    filler_pattern = re.compile(r"|".join(filler_words), re.IGNORECASE)
    text = subtitle_text.replace("\n", " ")
    text = filler_pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_subtitles(filepath, len_markers=15):
    with Path.open(filepath, "r") as file:
        data = json.load(file)

    for item in data:
        item["text"] = clean_subtitle_text(item["text"])
        item["T0"] = {
            "word": item["text"][:len_markers],  # First `len_markers` characters
            "timestamp": item["start"],
        }
        item["T1"] = {
            "word": item["text"][-len_markers:],  # Last `len_markers` characters
            "timestamp": item["end"],
        }

        # Remove "start" and "end" from each item
        del item["start"]
        del item["end"]

    merged_subtitles = recursive_merge_subtitles(data)
    return merged_subtitles


def overlapping_chunks(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        # Ensure the start does not cut in the middle of a word
        if start > 0:
            # Find the last space before the start to avoid cutting off in the middle of a word
            start = text.rfind(" ", 0, start) + 1
            if start == -1:  # If no space found, start from the beginning
                start = 0

        # Determine the end of the chunk
        end = start + chunk_size

        # Ensure the end does not cut off in the middle of a word
        if end < len(text):
            # Find the last space before the end of the chunk
            end = text.rfind(" ", start, end)
            if end == -1:  # If no space found, just use the chunk size (handle edge case)
                end = start + chunk_size

        # Add the chunk
        chunks.append(text[start:end])

        # If we're at the end of the text, break
        if end == len(text):
            break

        # Move the start to the overlap position (next chunk starts at the last word of the previous chunk)
        start = end - overlap

    return chunks


def clean_sentences(sent_lst):
    cleaned = []
    for sentence in sent_lst:
        # Remove punctuation using str.translate
        no_punct = sentence.translate(str.maketrans("", "", string.punctuation))
        cleaned.append(no_punct)
    return cleaned


def clean_decoded_text(text):
    text = re.sub(r"\b(\w+)(\s+\1){2,}", r"\1", text)  # remove repeated words
    text = re.sub(r"\.{3,}", "...", text)  # limit long ellipses
    text = re.sub(r"(?:\s*\.\.\.){2,}", "...", text)  # collapse multiple ellipses
    text = re.sub(r"\s+", " ", text).strip()  # normalize spacing
    return text


def separate_into_chunks(combined_subtitles, sents_lst, max_chars=300, time_jump_threshold=10):
    word_timings = [
        (v["word"], convert_to_float_seconds(v["timestamp"]))
        for k, v in combined_subtitles.items()
        if k.startswith("T")
    ]
    chunks = []
    current_chunk = []
    current_length = 0

    for sent_text in sents_lst:
        # Add the sentence to the current chunk
        current_chunk.append(sent_text)
        current_length += len(sent_text) + 1  # Adding 1 for the space between sentences

        # Check if adding this sentence exceeds the max length
        if current_length > max_chars:
            # Add the current chunk to the chunks list
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with the current sentence
            current_chunk = []
            current_length = 0

    # Add the last chunk if it's not empty
    if len(current_chunk) != 0:
        chunks.append(" ".join(current_chunk))

    subtitle_chunks = []
    last_used_ts = 0  # Initial starting point (0:00)

    for chunk in chunks:
        last_seen_ts = None
        matching = [(word, ts) for word, ts in word_timings if word in chunk and ts > last_used_ts]

        if matching:
            start_ts = min(ts for _, ts in matching)

            for match in matching:
                candidate_ts = match[1]

                if last_seen_ts is None:
                    last_seen_ts = candidate_ts
                    best_end_ts = candidate_ts
                elif candidate_ts - last_seen_ts <= time_jump_threshold:
                    last_seen_ts = candidate_ts
                    best_end_ts = candidate_ts
                # else: skip this timestamp, don't update best_end_ts

            end_ts = best_end_ts
            last_used_ts = end_ts if end_ts is not None else last_used_ts
        else:
            start_ts = -1
            end_ts = -1

        subtitle_chunks.append(
            {"start": float_to_timestamp_str(start_ts), "end": float_to_timestamp_str(end_ts), "text": chunk}
        )
    return subtitle_chunks


def process_subtitles_file(
    filepath,
    idx,
    tokenizer,
    model,
    nlp,
    output_dir,
    logger,
    repetition_penalty=2.5,
    max_new_tokens=256,
):
    try:
        logger.info("Processing: %s", filepath)
        combined_subtitles = combine_subtitles(filepath)
        chunks = overlapping_chunks(combined_subtitles["text"])

        decoded_chunks = []
        for chunk_text in chunks:
            inputs = tokenizer.encode("punctuate: " + chunk_text, return_tensors="tf", truncation=True)
            result = model.generate(inputs, repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens)
            decoded = tokenizer.decode(result[0], skip_special_tokens=True)
            cleaned = clean_decoded_text(decoded)
            decoded_chunks.append(cleaned)

        merged_chunks = recursive_merge_subtitle_texts(decoded_chunks)
        doc = nlp(merged_chunks)
        sent_lst = clean_sentences([sent.text.strip().lower() for sent in doc.sents])
        combined_subtitles["text"] = merged_chunks  # for timestamp alignment

        final_chunks = separate_into_chunks(combined_subtitles, sent_lst)
        json_path = output_dir / "subtitles_chunks_video_%d.json" % (idx + 1)
        with Path.open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        logger.info("✅ Saved chunked subtitles to %s", json_path)

    except Exception as e:
        logger.error("❌ Failed to process %s: %s", filepath, str(e))


@step
def chunk():
    logger = get_logger(__name__)

    # Load models
    tokenizer = T5Tokenizer.from_pretrained("SJ-Ray/Re-Punctuate")
    model = TFT5ForConditionalGeneration.from_pretrained("SJ-Ray/Re-Punctuate")
    nlp = spacy.load("en_core_web_sm")

    # Prepare output directory
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "subtitle_chunks_json"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("🗑️ Existing subtitle_chunks_json directory removed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("📂 Subtitle_chunks_json directory created at %s", output_dir)

    subtitles_dir = project_root / "subtitles_json"
    subtitle_files = [
        subtitles_dir / filename for filename in sorted(os.listdir(subtitles_dir)) if filename.endswith(".json")
    ]

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each subtitle file
        for idx, filepath in enumerate(subtitle_files):
            executor.submit(process_subtitles_file, filepath, idx, tokenizer, model, nlp, output_dir, logger)


chunk()
