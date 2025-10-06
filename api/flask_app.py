import os
import time
import psutil
import numpy as np
import tempfile
import shutil
import subprocess
from urllib.parse import urlparse

# --- NumPy compatibility shim (for older libs like resemblyzer) ---
# NumPy >= 1.24 removed np.bool, np.int, np.float; some libs still reference them.
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
# ------------------------------------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import after the NumPy shim so resemblyzer sees the aliases
from resemblyzer import VoiceEncoder, preprocess_wav

import requests
import librosa
import soundfile as sf

# Use a self-contained ffmpeg (works on Replit; no sudo needed)
import imageio_ffmpeg
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()  # full path to bundled ffmpeg

app = Flask(__name__)

# CORS
CORS(
    app,
    origins=["*"],
    methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Config
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "flac", "ogg", "webm"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Speaker encoder
encoder = VoiceEncoder()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_audio_format(path: str) -> str:
    _, ext = os.path.splitext(path.lower())
    return ext[1:] if ext else "unknown"

def is_valid_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return bool(u.scheme and u.netloc)
    except Exception:
        return False

def get_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    fname = os.path.basename(parsed.path)
    if not fname or "." not in fname:
        fname = f"audio_{int(time.time())}.webm"
    return fname

def download_audio_from_url(url: str, out_path: str) -> str:
    try:
        print(f"Downloading audio from URL: {url}")
        headers = {
            "User-Agent": "VoiceSimilarityMatcher/1.0",
            "Accept": "audio/*,*/*;q=0.9",
        }
        with requests.get(url, headers=headers, timeout=30, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded audio to: {out_path}")
        return out_path
    except Exception as e:
        raise Exception(f"Failed to download audio from URL: {e}")

def convert_to_wav_ffmpeg(input_path: str, output_path: str) -> str:
    """
    Convert with bundled ffmpeg. Handles webm/ogg/mp3/m4a/flac reliably.
    """
    try:
        if not FFMPEG_BIN or not os.path.exists(FFMPEG_BIN):
            print("Bundled ffmpeg not found, falling back to librosa")
            return convert_to_wav_librosa(input_path, output_path)

        print(f"Converting {input_path} -> {output_path} using ffmpeg at {FFMPEG_BIN}")

        cmd = [
            FFMPEG_BIN,
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", input_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("ffmpeg stderr:", result.stderr)
            raise RuntimeError("ffmpeg failed")

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("ffmpeg produced empty output")

        print(f"ffmpeg conversion successful: {output_path}")
        return output_path
    except Exception as e:
        print(f"ffmpeg conversion failed: {e}")
        print("Falling back to librosa")
        return convert_to_wav_librosa(input_path, output_path)

def convert_to_wav_librosa(input_path: str, output_path: str) -> str:
    """
    Fallback via librosa/soundfile.
    NOTE: libsndfile cannot decode webm/ogg; raise early in that case.
    """
    ext = os.path.splitext(input_path.lower())[1]
    if ext in [".webm", ".ogg"]:
        raise Exception("librosa/soundfile cannot decode WebM/Ogg without ffmpeg.")

    try:
        print(f"Converting {input_path} -> {output_path} using librosa")
        audio_data, sample_rate = librosa.load(input_path, sr=16000, mono=True)
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
        print(f"librosa conversion successful: {output_path}")
        return output_path
    except Exception as e:
        raise Exception(f"Error converting audio with librosa: {e}")

def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    Preferred path: ffmpeg; fallback: librosa (non-webm/ogg).
    """
    try:
        return convert_to_wav_ffmpeg(input_path, output_path)
    except Exception as e:
        print(f"All conversion methods failed: {e}")
        raise Exception(f"Error converting audio: {e}")

def analyze_voice_similarity(audio_file1_path: str, audio_file2_path: str) -> dict:
    try:
        print(f"Processing audio files: {audio_file1_path}, {audio_file2_path}")

        print("Preprocessing audio file 1...")
        wav1 = preprocess_wav(audio_file1_path)
        print(f"Audio 1 shape: {wav1.shape}")

        print("Preprocessing audio file 2...")
        wav2 = preprocess_wav(audio_file2_path)
        print(f"Audio 2 shape: {wav2.shape}")

        print("Extracting embeddings...")
        embed1 = encoder.embed_utterance(wav1)
        embed2 = encoder.embed_utterance(wav2)

        similarity = float(
            np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
        )

        threshold = 0.80
        is_same_person = similarity >= threshold
        result = "SAME PERSON" if is_same_person else "DIFFERENT PEOPLE"

        print(f"Similarity score: {similarity}")

        return {
            "similarity_score": similarity,
            "is_same_person": bool(is_same_person),
            "conclusion": result,
            "threshold": threshold,
        }
    except Exception as e:
        print(f"Error in analyze_voice_similarity: {e}")
        raise Exception(f"Error processing audio files: {e}")

@app.route("/compare_voices", methods=["POST"])
def compare_voices():
    try:
        data = request.get_json()
        if not data or "audio1_url" not in data or "audio2_url" not in data:
            return jsonify({"error": "Both audio1_url and audio2_url are required in JSON request"}), 400

        audio1_url = data["audio1_url"]
        audio2_url = data["audio2_url"]

        if not is_valid_url(audio1_url) or not is_valid_url(audio2_url):
            return jsonify({"error": "Invalid URLs provided"}), 400

        fn1 = get_filename_from_url(audio1_url)
        fn2 = get_filename_from_url(audio2_url)

        if not (allowed_file(fn1) and allowed_file(fn2)):
            return jsonify({"error": f'Invalid file type in URLs. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{fn1}") as t1:
            temp1_path = t1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{fn2}") as t2:
            temp2_path = t2.name

        download_audio_from_url(audio1_url, temp1_path)
        download_audio_from_url(audio2_url, temp2_path)

        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_mem = process.memory_info().rss / (1024 * 1024)

        converted_files = []
        final_path1 = temp1_path
        final_path2 = temp2_path

        try:
            fmt1 = get_audio_format(temp1_path)
            if fmt1 != "wav":
                wav1_path = temp1_path.replace(f".{fmt1}", ".wav")
                convert_to_wav(temp1_path, wav1_path)
                converted_files.append(wav1_path)
                final_path1 = wav1_path

            fmt2 = get_audio_format(temp2_path)
            if fmt2 != "wav":
                wav2_path = temp2_path.replace(f".{fmt2}", ".wav")
                convert_to_wav(temp2_path, wav2_path)
                converted_files.append(wav2_path)
                final_path2 = wav2_path

            result = analyze_voice_similarity(final_path1, final_path2)

            exec_time = time.time() - start_time
            final_mem = process.memory_info().rss / (1024 * 1024)
            mem_used = final_mem - initial_mem

            result.update(
                {
                    "execution_time_seconds": round(exec_time, 4),
                    "memory_usage_mb": round(mem_used, 2),
                    "status": "success",
                }
            )
            return jsonify(result)
        finally:
            # cleanup
            for p in [temp1_path, temp2_path, *converted_files]:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except OSError:
                    pass

    except Exception as e:
        print(f"Error in compare_voices endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Voice matching API is running"})

@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "Voice Matching API",
            "version": "2.1",
            "endpoints": {
                "POST /compare_voices": "Compare two voice audio files from URLs",
                "GET /health": "Health check",
                "GET /": "API documentation",
            },
            "usage": {
                "method": "POST",
                "endpoint": "/compare_voices",
                "content_type": "application/json",
                "parameters": {
                    "audio1_url": "First audio file URL (wav, mp3, m4a, flac, ogg, webm)",
                    "audio2_url": "Second audio file URL (wav, mp3, m4a, flac, ogg, webm)",
                },
            },
        }
    )

# WSGI app for Vercel
app = app

if __name__ == "__main__":
    # Helpful startup logs
    try:
        ver = subprocess.check_output([FFMPEG_BIN, "-version"], text=True).splitlines()[0]
        print("Using ffmpeg:", ver, "at", FFMPEG_BIN)
    except Exception as e:
        print("ffmpeg not available at startup:", e)
    app.run(debug=False, host="0.0.0.0", port=5000)
