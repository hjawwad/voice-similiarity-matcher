import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import gc

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=['*'], methods=['GET', 'POST'], allow_headers=['Content-Type'])

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simple_voice_analysis(audio_file1_path, audio_file2_path):
    """Simple voice analysis based on file properties."""
    try:
        print(f"Processing audio files: {audio_file1_path}, {audio_file2_path}")
        
        # Get file sizes as a simple comparison metric
        size1 = os.path.getsize(audio_file1_path)
        size2 = os.path.getsize(audio_file2_path)
        
        # Simple similarity based on file size ratio
        size_ratio = min(size1, size2) / max(size1, size2)
        
        # Mock similarity score (in real implementation, this would be actual audio analysis)
        similarity = size_ratio * 0.8  # Scale down for realistic values
        
        # Determine if voices are from the same source
        is_same_person = similarity >= 0.70
        result = "SAME PERSON" if is_same_person else "DIFFERENT PEOPLE"
        
        print(f"Similarity score: {similarity}")
        
        return {
            'similarity_score': float(similarity),
            'is_same_person': bool(is_same_person),
            'conclusion': result,
            'threshold': 0.70,
            'method': 'file_properties',
            'note': 'This is a demo implementation. For production, use proper audio analysis.'
        }
    except Exception as e:
        print(f"Error in simple_voice_analysis: {str(e)}")
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
            # Analyze voice similarity
            result = simple_voice_analysis(temp1_path, temp2_path)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            
            # Add performance metrics to result
            result.update({
                'execution_time_seconds': round(execution_time, 4),
                'status': 'success'
            })
            
            return jsonify(result)
            
        finally:
            # Clean up temporary files and memory
            try:
                os.unlink(temp1_path)
                os.unlink(temp2_path)
            except OSError:
                pass  # Files might already be deleted
            
            # Force garbage collection
            gc.collect()
    
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
                'similarity_score': 'Similarity score (0-1)',
                'is_same_person': 'Boolean indicating if same person',
                'conclusion': 'Human readable result',
                'threshold': 'Similarity threshold used (0.70)',
                'method': 'File properties analysis',
                'execution_time_seconds': 'Processing time',
                'status': 'success/error'
            }
        }
    })

# WSGI application for Vercel
app = app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)