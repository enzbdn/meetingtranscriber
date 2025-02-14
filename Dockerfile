FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p temp

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV ASR_MODEL="openai/whisper-base"
ENV DIARIZATION_MODEL="pyannote/speaker-diarization@2.1"
ENV OLLAMA_API_URL="http://localhost:11434/api/generate"
ENV OLLAMA_MODEL="llama3.1"

# Expose port
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]