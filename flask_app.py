import os
import time
import psutil
import numpy as np
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from resemblyzer import VoiceEncoder, preprocess_wav
import tempfile

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=['*'], methods=['GET', 'POST'], allow_headers=['Content-Type'])

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the VoiceEncoder
encoder = VoiceEncoder()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format using FFmpeg."""
    try:
        # Use FFmpeg to convert to WAV format
        cmd = [
            'ffmpeg',
            '-i', input_path,           # Input file
            '-acodec', 'pcm_s16le',    # PCM 16-bit little-endian
            '-ar', '16000',            # Sample rate 16kHz
            '-ac', '1',                
            '-y',                     
            output_path                
        ]
        
        print(f"Converting {input_path} to {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")
        
        print(f"Conversion successful: {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Audio conversion timed out")
    except Exception as e:
        raise Exception(f"Error converting audio: {str(e)}")

def get_audio_format(file_path):
    """Detect audio format from file extension."""
    _, ext = os.path.splitext(file_path.lower())
    return ext[1:] if ext else 'unknown'

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
    """API endpoint to compare two voice audio files."""
    try:
        # Check if both files are present in the request
        if 'audio1' not in request.files or 'audio2' not in request.files:
            return jsonify({
                'error': 'Both audio1 and audio2 files are required'
            }), 400
        
        audio1 = request.files['audio1']
        audio2 = request.files['audio2']
        
        # Check if files are selected
        if audio1.filename == '' or audio2.filename == '':
            return jsonify({
                'error': 'No files selected'
            }), 400
        
        # Check if files have allowed extensions
        if not (allowed_file(audio1.filename) and allowed_file(audio2.filename)):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Record start time for performance metrics
        start_time = time.time()
        
        # Get current process information for memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # in MB
        
        # Save uploaded files temporarily
        filename1 = secure_filename(audio1.filename)
        filename2 = secure_filename(audio2.filename)
        
        # Use temporary files for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{filename1}') as temp1:
            audio1.save(temp1.name)
            temp1_path = temp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{filename2}') as temp2:
            audio2.save(temp2.name)
            temp2_path = temp2.name
        
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
        'version': '1.0',
        'endpoints': {
            'POST /compare_voices': 'Compare two voice audio files',
            'GET /health': 'Health check',
            'GET /': 'API documentation'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/compare_voices',
            'content_type': 'multipart/form-data',
            'parameters': {
                'audio1': 'First audio file (wav, mp3, m4a, flac, ogg)',
                'audio2': 'Second audio file (wav, mp3, m4a, flac, ogg)'
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

