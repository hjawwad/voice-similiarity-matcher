import os
import time
import psutil
import numpy as np
import gradio as gr
from resemblyzer import VoiceEncoder, preprocess_wav

# Initialize the VoiceEncoder
encoder = VoiceEncoder()

def analyze_voice_similarity(audio_file1, audio_file2):
    start_time = time.time()  # Record start time
    
    # Get current process information
    process = psutil.Process(os.getpid())
    
    # Preprocess audio files
    try:
        wav1 = preprocess_wav(audio_file1)
        wav2 = preprocess_wav(audio_file2)
    except Exception as e:
        return f"Error processing audio files: {str(e)}"
    
    # Extract speaker embeddings
    embed1 = encoder.embed_utterance(wav1)
    embed2 = encoder.embed_utterance(wav2)
    
    # Calculate cosine similarity between embeddings
    similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
    
    # Determine if voices are from the same source
    result = "SAME PERSON" if similarity >= 0.80 else "DIFFERENT PEOPLE"
    
    # Get memory usage
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Format the output
    output = f"""
### Voice Similarity Analysis Results

**Similarity Score**: {similarity:.4f}
**Conclusion**: {result}

### Performance Metrics
**Memory Usage**: {memory_usage:.2f} MB
**Execution Time**: {execution_time:.4f} seconds
    """
    
    return output

# Create Gradio interface
with gr.Blocks(title="Voice Similarity Analyzer") as demo:
    gr.Markdown("# 🎤 Voice Similarity Analyzer")
    gr.Markdown("Upload two audio files to check if they're from the same person. A similarity score >= 0.80 indicates the same speaker.")
    
    with gr.Row():
        with gr.Column():
            audio_input1 = gr.Audio(label="Voice Sample 1", type="filepath")
        with gr.Column():
            audio_input2 = gr.Audio(label="Voice Sample 2", type="filepath")
    
    analyze_button = gr.Button("Analyze Voice Similarity", variant="primary")
    output_text = gr.Markdown(label="Results")
    
    analyze_button.click(
        fn=analyze_voice_similarity,
        inputs=[audio_input1, audio_input2],
        outputs=output_text
    )
    
    gr.Markdown("""
    ## How It Works
    1. Upload two voice recordings
    2. Click "Analyze Voice Similarity"
    3. The app will extract voice embeddings using Resemblyzer
    4. The similarity score is calculated using cosine similarity
    5. A score >= 0.80 indicates the same speaker

    ## Performance Metrics
    - Memory usage shows how much RAM is being used
    - Execution time measures how long the comparison takes
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()