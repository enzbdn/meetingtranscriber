import os
import logging
from flask import Flask, render_template, request
from pyannote.audio import Pipeline
import requests
from config import Config
from transcription_service import TranscriptionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize services
diarization_pipeline = Pipeline.from_pretrained(
    Config.DIARIZATION_MODEL,
    use_auth_token=True
)

def remote_asr_pipeline(audio_path):
    """Call remote ASR server API"""
    with open(audio_path, 'rb') as audio_file:
        response = requests.post(
            Config.WHISPER_API_URL,
            files = {'audio_file': audio_file},
            params={'output': 'json'},
            timeout=360
        )
        response.raise_for_status()
        return response.json().get('text', '')

asr_pipeline = remote_asr_pipeline

transcription_service = TranscriptionService(
    diarization_pipeline,
    asr_pipeline,
    Config.TEMP_DIR
)

def format_transcription(segments):
    """Format transcription results into readable text"""
    formatted = []
    for segment in segments:
        timestamp = f"[{segment['timestamp'][0]:.1f}s - {segment['timestamp'][1]:.1f}s]"
        formatted.append(f"{segment['speaker']} {timestamp}:\n{segment['text']}")
    return "\n\n".join(formatted)

def format_with_ollama(text):
    """Format transcription using Ollama LLM"""
    try:
        response = requests.post(
            Config.OLLAMA_API_URL,
            json={
            'model': Config.OLLAMA_MODEL,
            'prompt': f"Convert this transcription into well-formatted markdown. Only provide the markdown content:\n{text}",
            'stream': False
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        logger.error("Ollama API error: %s", str(e))
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    try:
        if 'audio_file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['audio_file']
        if not file or not file.filename:
            return render_template('index.html', error='No file selected')

        if not Config.allowed_file(file.filename):
            return render_template('index.html', error='Invalid file type')

        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        input_path = os.path.join(Config.TEMP_DIR, "input." + file.filename.rsplit('.', 1)[1])
        wav_path = os.path.join(Config.TEMP_DIR, "audio.wav")

        try:
            file.save(input_path)
            if input_path.endswith('.wav'):
                wav_path = input_path
            else:
                transcription_service.convert_to_wav(input_path, wav_path)

            results = transcription_service.process_audio(wav_path)
            raw_transcription = format_transcription(results)
            markdown_transcription = format_with_ollama(raw_transcription)
            return render_template('index.html', 
                                transcription=markdown_transcription,
                                raw_transcription=raw_transcription)

        finally:
            for path in [input_path, wav_path]:
                if os.path.exists(path):
                    os.remove(path)

    except Exception as e:
        logger.error("Error: %s", str(e))
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)