import os
import time
import psutil
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from resemblyzer import VoiceEncoder, preprocess_wav
import tempfile
import librosa
import soundfile as sf
import requests
from urllib.parse import urlparse

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=['*'], methods=['GET', 'POST'], allow_headers=['Content-Type'])

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Initialize the VoiceEncoder
encoder = VoiceEncoder()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format using librosa."""
    try:
        print(f"Converting {input_path} to {output_path}")
        
        # Load audio file with librosa
        audio_data, sample_rate = librosa.load(input_path, sr=16000, mono=True)
        
        # Save as WAV file
        sf.write(output_path, audio_data, sample_rate, subtype='PCM_16')
        
        print(f"Conversion successful: {output_path}")
        return output_path
        
    except Exception as e:
        raise Exception(f"Error converting audio: {str(e)}")

def get_audio_format(file_path):
    """Detect audio format from file extension."""
    _, ext = os.path.splitext(file_path.lower())
    return ext[1:] if ext else 'unknown'

def download_audio_from_url(url, output_path):
    """Download audio file from URL and save to local path."""
    try:
        print(f"Downloading audio from URL: {url}")
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'VoiceSimilarityMatcher/1.0',
            'Accept': 'audio/*,*/*;q=0.9'
        }
        
        # Download the file
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded audio to: {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download audio from URL: {str(e)}")
    except Exception as e:
        raise Exception(f"Error downloading audio: {str(e)}")

def get_filename_from_url(url):
    """Extract filename from URL."""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename or '.' not in filename:
        # Generate a default filename with extension
        filename = f"audio_{int(time.time())}.webm"
    return filename

def is_valid_url(url):
    """Check if the provided string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def analyze_voice_similarity(audio_file1_path, audio_file2_path):
    """Analyze voice similarity between two audio files."""
    try:
        print(f"Processing audio files: {audio_file1_path}, {audio_file2_path}")
        
        # Preprocess audio files
        print("Preprocessing audio file 1...")
        wav1 = preprocess_wav(audio_file1_path)
        print(f"Audio 1 shape: {wav1.shape}")
        
        print("Preprocessing audio file 2...")
        wav2 = preprocess_wav(audio_file2_path)
        print(f"Audio 2 shape: {wav2.shape}")
        
        # Extract speaker embeddings
        print("Extracting embeddings...")
        embed1 = encoder.embed_utterance(wav1)
        embed2 = encoder.embed_utterance(wav2)
        
        # Calculate cosine similarity between embeddings
        similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
        
        # Determine if voices are from the same source
        is_same_person = similarity >= 0.80
        result = "SAME PERSON" if is_same_person else "DIFFERENT PEOPLE"
        
        print(f"Similarity score: {similarity}")
        
        return {
            'similarity_score': float(similarity),
            'is_same_person': bool(is_same_person),
            'conclusion': result,
            'threshold': 0.80
        }
    except Exception as e:
        print(f"Error in analyze_voice_similarity: {str(e)}")
        raise Exception(f"Error processing audio files: {str(e)}")

@app.route('/compare_voices', methods=['POST'])
def compare_voices():
    """API endpoint to compare two voice audio files from URLs."""
    try:
        # Expect JSON data with URLs
        data = request.get_json()
        
        if not data or 'audio1_url' not in data or 'audio2_url' not in data:
            return jsonify({
                'error': 'Both audio1_url and audio2_url are required in JSON request'
            }), 400
        
        audio1_url = data['audio1_url']
        audio2_url = data['audio2_url']
        
        # Validate URLs
        if not is_valid_url(audio1_url) or not is_valid_url(audio2_url):
            return jsonify({
                'error': 'Invalid URLs provided'
            }), 400
        
        # Extract filenames from URLs
        filename1 = get_filename_from_url(audio1_url)
        filename2 = get_filename_from_url(audio2_url)
        
        # Check if URLs have allowed extensions
        if not (allowed_file(filename1) and allowed_file(filename2)):
            return jsonify({
                'error': f'Invalid file type in URLs. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Create temporary files for downloaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{filename1}') as temp1:
            temp1_path = temp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{filename2}') as temp2:
            temp2_path = temp2.name
        
        # Download audio files from URLs
        download_audio_from_url(audio1_url, temp1_path)
        download_audio_from_url(audio2_url, temp2_path)
        
        # Record start time for performance metrics
        start_time = time.time()
        
        # Get current process information for memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # in MB
        
        try:
            # Convert audio files to WAV if needed
            converted_files = []
            
            # Check and convert first audio file
            audio_format1 = get_audio_format(temp1_path)
            if audio_format1 != 'wav':
                wav1_path = temp1_path.replace(f'.{audio_format1}', '.wav')
                convert_to_wav(temp1_path, wav1_path)
                converted_files.append(wav1_path)
                final_path1 = wav1_path
            else:
                final_path1 = temp1_path
            
            # Check and convert second audio file
            audio_format2 = get_audio_format(temp2_path)
            if audio_format2 != 'wav':
                wav2_path = temp2_path.replace(f'.{audio_format2}', '.wav')
                convert_to_wav(temp2_path, wav2_path)
                converted_files.append(wav2_path)
                final_path2 = wav2_path
            else:
                final_path2 = temp2_path
            
            # Analyze voice similarity with converted files
            result = analyze_voice_similarity(final_path1, final_path2)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / (1024 * 1024)  # in MB
            memory_usage = final_memory - initial_memory
            
            # Add performance metrics to result
            result.update({
                'execution_time_seconds': round(execution_time, 4),
                'memory_usage_mb': round(memory_usage, 2),
                'status': 'success'
            })
            
            return jsonify(result)
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp1_path)
                os.unlink(temp2_path)
                # Clean up converted files
                for converted_file in converted_files:
                    os.unlink(converted_file)
            except OSError:
                pass  # Files might already be deleted
    
    except Exception as e:
        print(f"Error in compare_voices endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Voice matching API is running'
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation."""
    return jsonify({
        'message': 'Voice Matching API',
        'version': '2.0',
        'endpoints': {
            'POST /compare_voices': 'Compare two voice audio files from URLs',
            'GET /health': 'Health check',
            'GET /': 'API documentation'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/compare_voices',
            'content_type': 'application/json',
            'parameters': {
                'audio1_url': 'First audio file URL (wav, mp3, m4a, flac, ogg, webm)',
                'audio2_url': 'Second audio file URL (wav, mp3, m4a, flac, ogg, webm)'
            },
            'example': {
                'audio1_url': 'https://example.com/audio1.webm',
                'audio2_url': 'https://example.com/audio2.webm'
            },
            'response': {
                'similarity_score': 'Cosine similarity score (0-1)',
                'is_same_person': 'Boolean indicating if same person',
                'conclusion': 'Human readable result',
                'threshold': 'Similarity threshold used (0.80)',
                'execution_time_seconds': 'Processing time',
                'memory_usage_mb': 'Memory used during processing',
                'status': 'success/error'
            }
        }
    })

# WSGI application for Vercel
app = app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)

