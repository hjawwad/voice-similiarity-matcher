# Voice Similarity Analyzer - Replit Project

## Overview

This is a Voice Matching API project with a Gradio-based web interface for comparing voice similarity using Resemblyzer embeddings. The application analyzes two voice audio samples and determines if they're from the same speaker using cosine similarity.

**Current Status**: Fully configured and running on Replit

## Project Architecture

### Main Components

1. **Gradio Frontend (app.py)** - Primary web interface
   - Interactive web UI for voice similarity analysis
   - Runs on port 5000
   - Uses Resemblyzer for voice embeddings
   - Configured for Replit environment with allowed_paths=["*"] for proxy support

2. **Flask API (api/flask_app.py)** - Backend REST API
   - Located in `/api/` directory
   - Originally designed for Vercel deployment
   - Provides REST endpoints for voice comparison
   - Currently uses simplified file-based comparison (mock implementation)

3. **HTML Test Interface (test_interface.html)** - Static testing tool
   - Browser-based testing interface
   - Includes voice recording capabilities
   - For testing the Flask API

### Key Technologies

- **Python 3.12**
- **Gradio 5.47+** - Web UI framework
- **Resemblyzer 0.1.4** - Voice embedding extraction
- **PyTorch (CPU-only)** - Deep learning backend
- **Librosa** - Audio processing
- **NumPy, SciPy** - Numerical computing
- **Flask + Flask-CORS** - REST API (optional)

## Recent Changes

### 2025-09-30: Initial Replit Setup
- Installed all Python dependencies including Gradio, Resemblyzer, and PyTorch (CPU version)
- Configured app.py to run on port 5000 with host 0.0.0.0
- Added allowed_paths=["*"] to support Replit's proxy/iframe setup
- Set up workflow "Gradio Voice Analyzer" to run the frontend
- Updated .gitignore with Replit-specific entries (.pythonlibs/, .upm/, .cache/)
- Configured deployment for autoscale using the Gradio app

## Configuration

### Environment Setup
- Language: Python 3.12 (via Nix modules)
- Package Manager: pip (installed packages in .pythonlibs/)
- Port: 5000 (frontend)
- Host: 0.0.0.0 (required for Replit)

### Workflow
- **Name**: Gradio Voice Analyzer
- **Command**: `python app.py`
- **Port**: 5000
- **Type**: webview

### Deployment
- **Target**: autoscale
- **Run Command**: `["python", "app.py"]`
- Suitable for stateless web applications
- No build step required

## How It Works

1. User uploads two audio files (WAV, MP3, M4A, FLAC, OGG, WEBM)
2. Files are preprocessed using Resemblyzer
3. Voice embeddings are extracted using VoiceEncoder
4. Cosine similarity is calculated between embeddings
5. Similarity score >= 0.80 indicates same speaker
6. Results include:
   - Similarity score
   - Conclusion (SAME PERSON / DIFFERENT PEOPLE)
   - Memory usage
   - Execution time

## File Structure

```
Voice-matching/
├── app.py                     # Gradio frontend (PRIMARY)
├── api/
│   └── flask_app.py          # Flask API backend
├── requirements.txt           # Flask API dependencies
├── requirements-gradio.txt    # Gradio app dependencies
├── test_interface.html        # HTML testing interface
├── vercel.json               # Vercel deployment config
├── .replit                   # Replit configuration
├── .gitignore               # Git ignore rules
└── replit.md                # This file
```

## Known Issues & Notes

- PyTorch CPU version is used to save disk space (no CUDA dependencies)
- The Flask API in `/api/` uses a simplified mock implementation
- For production voice analysis, the Gradio app with Resemblyzer is recommended
- Voice encoder model loads on CPU (takes ~0.02 seconds)

## Future Enhancements

- Integrate actual Resemblyzer model into Flask API
- Add sample audio files for testing
- Implement batch voice comparison
- Add voice recording directly in Gradio interface
- Support for longer audio files

## User Preferences

None specified yet.
