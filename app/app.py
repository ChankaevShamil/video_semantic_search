import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import logging
import av
import yt_dlp
import gradio as gr

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import faiss


processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

DATA_DIR = Path("/app/data")
VIDEO_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
INDEX_DIR = DATA_DIR / "indices"
INDEX_CACHE = {}
CACHE_MAX_SIZE = 5
EMBEDDING_DIM = model.config.projection_dim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CLIP_API_URL = os.getenv("CLIP_API_URL")
if not CLIP_API_URL:
    logging.error("CLIP_API_URL isnt installed in venv!")


# ----------Load video from youtube --------
def get_youtube_video_id(url):
    """
    Any youtube video have unique ID from 11 characters: [0-9|A-Z|a-z|_|-].
    This ID will use as video ID in our program.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('youtu.be',):
        return parsed_url.path[1:]
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
        if parsed_url.path == '/watch':
            p = parse_qs(parsed_url.query)
            return p.get('v', [None])[0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[1]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/v/')[1]
    return None

def download_video(youtube_url: str) -> Path | None:
    """
    Get video url and dowload it in data/videos
    """"

    video_id = get_youtube_video_id(youtube_url)
    output_template = VIDEO_DIR / f"{video_id}.%(ext)s"

    existing_videos = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if existing_videos:
        logging.info(f"Video {video_id} exist: {existing_videos[0]}")
        return existing_videos[0]

    # download best version of video for better result in video search
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': str(output_template),
        'noplaylist': True,
        'quiet': True,
    }

    logging.info(f"Start loading video: {youtube_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    downloaded_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if downloaded_files:
        logging.info(f"Video loaded: {downloaded_files[0]}")
        return downloaded_files[0]


# ---------- Extract I-frames --------
def extract_keyframes(video_path: Path) -> list[tuple[float, Path]] | None:
    """
    mp4 format - is specific compressed video format.
    Instead, load all frames of video, this format contain only first frame
    and diff for the next frames. So, we can load only this specific I-frames of all video.
    5 min video have 50-100 I-frames.
    """
    video_id = video_path.stem
    video_frames_dir = FRAMES_DIR / video_id
    video_frames_dir.mkdir(parents=True, exist_ok=True)

    keyframes_info_file = video_frames_dir / "keyframes_timestamps_pyav.json"

    logging.info(f"I-frames extracting from: {video_path}")
    keyframes_data = []
    frame_counter = 0

    # there is a lot of variants to get I-frames
    # most simple variante - using ffmpeg.
    # but this package work badly on windows, so we use av module
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base or container.time_base

        logging.info(
            f"Video {video_id}: time_base={time_base}, duration={stream.duration * time_base if stream.duration else 'N/A'}s")

        for packet in container.demux(stream):
            for frame in packet.decode():
                if not frame.key_frame:
                    continue
                timestamp_sec = float(frame.pts * time_base)

                frame_counter += 1
                frame_filename = f"keyframe_{frame_counter:06d}.jpg"
                frame_path = video_frames_dir / frame_filename

                img = frame.to_image()
                img.save(frame_path, quality=90)
                keyframes_data.append((timestamp_sec, frame_path))
                if frame_counter % 20 == 0:
                    logging.info(
                        f"Extracting {frame_counter} I-frames by (ts: {timestamp_sec:.3f}s)...")


    with open(keyframes_info_file, 'w') as f:
        json.dump([[ts, frame_path.name] for ts, frame_path in keyframes_data], f)
    logging.info(f"Extracting {len(keyframes_data)} I-frames for {video_id}.")
    return keyframes_data


# ---------- Create vector database --------
def get_local_embedding(data: Image.Image | str, input_type: str) -> np.ndarray | None:
    with torch.no_grad():
        if input_type == "image":
            if data.mode != "RGB":
                data = data.convert("RGB")
            inputs = processor(images=data, return_tensors="pt", padding=True, truncation=True).to(device)
            embeddings = model.get_image_features(**inputs)
        elif input_type == "text":
            inputs = processor(text=data, return_tensors="pt", padding=True, truncation=True).to(device)
            embeddings = model.get_text_features(**inputs)

        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        embedding_np = embeddings.squeeze().cpu().detach().numpy()

        return embedding_np.astype('float32')

def build_faiss_index(keyframes_data: list[tuple[float, Path]], video_id: str) -> tuple[faiss.Index, list[float]]:

    index_path = INDEX_DIR / f"{video_id}.index"
    map_path = INDEX_DIR / f"{video_id}.map.json"

    logging.info(
        f"Create FAISS database for {video_id}")

    embeddings_list = []
    timestamp_map = []
    processed_indices = set()

    for i, (timestamp, frame_path) in enumerate(keyframes_data):
        logging.debug(f"Loading {i + 1}/{len(keyframes_data)}: {frame_path} (ts: {timestamp:.3f}s)")

        img = Image.open(frame_path)
        embedding_np = get_local_embedding(img, input_type="image")
        img.close()

        embeddings_list.append(embedding_np)
        timestamp_map.append(timestamp)
        processed_indices.add(i)


    logging.info(f"Get {len(embeddings_list)} embeddings. Creating FAISS...")
    embeddings_np_array = np.array(embeddings_list).astype('float32')

    # index = faiss.IndexFlatL2(EMBEDDING_DIM) # L2
    index = faiss.IndexFlatIP(EMBEDDING_DIM) # cos

    index.add(embeddings_np_array)

    logging.info(f"FAISS for {video_id} created. Total index size: {index.ntotal}")

    final_timestamp_map = [keyframes_data[i][0] for i in sorted(list(processed_indices))]

    logging.info(f"Saving FAISS in {index_path}")
    faiss.write_index(index, str(index_path))

    logging.info(f"Saving timestamps in {map_path}")
    with open(map_path, 'w') as f:
        json.dump(final_timestamp_map, f)

    return index, final_timestamp_map

def load_index_and_map(video_id: str) -> tuple[faiss.Index | None, list[float] | None]:

    index_path = INDEX_DIR / f"{video_id}.index"
    map_path = INDEX_DIR / f"{video_id}.map.json"

    index = faiss.read_index(str(index_path))
    with open(map_path, 'r') as f:
        timestamp_map = json.load(f)

    return index, timestamp_map


def search_in_video(video_id: str, text_query: str, k: int = 5) -> list[tuple[float, float]] | None | str:
    """
    Args:
        video_id: our video ID.
        text_query: user querys.
        k: number of returned results

    Returns:
        (timestamp, distance) | None | str (if index not found).
    """
    logging.info(f"Start searching in '{video_id}' by query: '{text_query}', k={k}")

    index, timestamp_map = load_index_and_map(video_id)
    query_embedding_np = get_local_embedding(text_query, input_type="text")
    query_np_2d = np.expand_dims(query_embedding_np, axis=0).astype('float32')


    actual_k = min(k, index.ntotal)
    distances, indices = index.search(query_np_2d, actual_k)

    if indices.size == 0 or indices[0][0] == -1:
        logging.info(f"Can't find '{text_query}' in {video_id}.")
        return []

    found_faiss_ids = indices[0]
    found_distances = distances[0]

    results = []
    for faiss_id, dist in zip(found_faiss_ids, found_distances):
        if faiss_id == -1:
            continue
        if 0 <= faiss_id < len(timestamp_map):
            results.append((timestamp_map[faiss_id], float(dist)))

    results.sort(key=lambda item: -item[1])

    logging.info(f"Found {len(results)} results.")
    if results:
        log_res = [f"{ts:.2f}s (dist: {d:.4f})" for ts, d in results]
        logging.info(f"Founded timestamps: {'; '.join(log_res)}")

    return results  # list[(timestamp, distance)]


# ------ Интерфейс ------

def format_timestamp(seconds: float) -> str:
    """Turn time in seconds to format HH:MM:SS.ms"""
    s = int(seconds)
    ms = int((seconds - s) * 1000)
    m = s // 60
    h = m // 60
    return f"{h:02}:{m % 60:02}:{s % 60:02}.{ms:03}"


def process_video_and_search(youtube_url: str, text_query: str, top_k: int = 5):
    output_messages = []
    video_display_path = None  # Путь к видео для отображения

    def update_status(message, current_video_path=None):
        nonlocal video_display_path
        if current_video_path:
            video_display_path = current_video_path
        output_messages.append(message)
        return "\n".join(output_messages), video_display_path

    downloaded_video_path = download_video(youtube_url)

    video_id = downloaded_video_path.stem
    logging.info(f"Video processing: {video_id}")

    video_display_path = str(downloaded_video_path)

    keyframes = extract_keyframes(downloaded_video_path)

    faiss_index, ts_map = build_faiss_index(keyframes, video_id)

    if faiss_index.ntotal == 0 and keyframes:
        yield update_status(
            f"Video'{video_id}' is indexed, but emedings not found.",
            video_display_path)
    elif faiss_index.ntotal == 0 and not keyframes:  # Если кадров не было и индекс пуст
        yield update_status(f"Video '{video_id}' hasn't I-frames.", video_display_path)
    else:
        yield update_status(f"Video '{video_id}' indexed. Total len: {faiss_index.ntotal}.",
                            video_display_path)

    yield update_status(f"Search video by query: '{text_query}'...", video_display_path)

    search_results_with_dist = search_in_video(video_id, text_query, k=top_k)

    if isinstance(search_results_with_dist, str):
        return update_status(search_results_with_dist, video_display_path)
    elif not search_results_with_dist:  # empty list
        return update_status(f"Cant find I-frames by query '{text_query}'.", video_display_path)
    else:
        results_text_parts = [f"Top-{len(search_results_with_dist)} results for '{text_query}':"]

        formatted_timestamps = []
        for ts, dist in search_results_with_dist:
            formatted_ts = format_timestamp(ts)
            results_text_parts.append(f"- {formatted_ts}")
            formatted_timestamps.append((formatted_ts, int(ts)))

        results_text_parts.append("\nLinks for moments in YouTube:")
        logging.info(results_text_parts)
        base_yt_url = f"https://www.youtube.com/watch?v={video_id}&t="
        for formatted_ts, original_ts_seconds in formatted_timestamps:
            results_text_parts.append(f"- {formatted_ts} -> {base_yt_url}{original_ts_seconds}s")

        final_output_text = "\n".join(results_text_parts)

        output_messages.append(final_output_text)
        yield "\n".join(output_messages), video_display_path
        return "\n".join(output_messages), video_display_path


if __name__ == "__main__":
    iface = gr.Interface(
        fn=process_video_and_search,\
        inputs=[
            gr.Textbox(label="YouTube URL", placeholder="Paste YouTube video URL"),
            gr.Textbox(label="Query", placeholder="What you want find in video?"),
            gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Results (Top K)")
        ],
        outputs=[
            gr.Textbox(label="Results", lines=10),  # Увеличим количество строк
            gr.Video(label="Loaded videos")
        ],
        title="Video semantic search in YouTube",
        description="Paste YouTube video link and print, what you want find in video.",
        allow_flagging="never",
        examples=[
            ["https://www.youtube.com/watch?v=vfUrK9pFfUg", "bridge", 6],
            ["https://www.youtube.com/watch?v=y_ZN_2a07hA", "a waterfall", 5],
            ["https://www.youtube.com/watch?v=FTQbiNvZqaY", "cat playing", 5],
            ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "man dancing in front of window", 5],
        ],
    )

    gradio_server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    gradio_server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    iface.launch(server_name=gradio_server_name, server_port=gradio_server_port)
