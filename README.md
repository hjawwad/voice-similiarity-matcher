# Voice Matching API

A Flask-based API for comparing voice similarity using Resemblyzer embeddings.

## Features

- Voice similarity analysis using cosine similarity
- Support for multiple audio formats (WAV, MP3, M4A, FLAC, OGG, WEBM)
- RESTful API endpoints
- Performance metrics (execution time, memory usage)
- CORS enabled for web applications

## API Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `POST /compare_voices` - Compare two voice audio files

## Local Development

### Prerequisites

- Python 3.8+
- FFmpeg (for local development with Gradio app)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

```bash
python api/flask_app.py
```

The API will be available at `http://localhost:5001`

### Running the Gradio Interface

```bash
pip install -r requirements-gradio.txt
python app.py
```

The Gradio interface will be available at `http://localhost:7860`

## Vercel Deployment

This project is configured for deployment on Vercel.

### Prerequisites

- Vercel account
- Vercel CLI installed (`npm i -g vercel`)

### Deployment Steps

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

4. **Follow the prompts**:
   - Link to existing project or create new
   - Confirm project settings
   - Deploy

### Important Notes for Vercel

- **File Size Limit**: Maximum 16MB per audio file
- **Execution Time**: Limited to 30 seconds per request
- **Memory**: Limited to 1GB per function
- **No FFmpeg**: Audio conversion uses librosa instead
- **Stateless**: No persistent file storage

### Testing the Deployment

After deployment, test the API:

```bash
curl https://your-app.vercel.app/health
```

## Usage Examples

### Compare Voices via API

```bash
curl -X POST https://your-app.vercel.app/compare_voices \
  -F "audio1=@voice1.wav" \
  -F "audio2=@voice2.wav"
```

### Response Format

```json
{
  "similarity_score": 0.8542,
  "is_same_person": true,
  "conclusion": "SAME PERSON",
  "threshold": 0.80,
  "execution_time_seconds": 2.3456,
  "memory_usage_mb": 45.67,
  "status": "success"
}
```

## Project Structure

```
Voice-matching/
├── api/
│   └── flask_app.py          # Main Flask API
├── app.py                     # Gradio interface
├── requirements.txt           # API dependencies
├── requirements-gradio.txt    # Gradio dependencies
├── vercel.json               # Vercel configuration
├── test_api.py               # API testing script
├── test_interface.html       # Web testing interface
└── README.md                 # This file
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Audio Format Issues**: Supported formats are WAV, MP3, M4A, FLAC, OGG, WEBM
3. **Memory Issues**: Large audio files may cause memory errors
4. **Timeout Issues**: Processing very long audio files may timeout

### Vercel-Specific Issues

1. **Function Timeout**: Increase `maxDuration` in vercel.json if needed
2. **Memory Limit**: Optimize audio processing for smaller memory footprint
3. **Cold Starts**: First request may be slower due to model loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.