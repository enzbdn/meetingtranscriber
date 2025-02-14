import os
import logging
from typing import List, Dict
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, diarization_pipeline, asr_pipeline, temp_dir: str):
        self.diarization_pipeline = diarization_pipeline
        self.asr_pipeline = asr_pipeline
        self.temp_dir = temp_dir
        self.chunk_length = 30

    def process_audio(self, audio_path: str) -> List[Dict]:
        """Process audio file and return transcription segments"""
        try:
            diarization = self.diarization_pipeline(
                audio_path,
                num_speakers=None,
                min_speakers=1,
                max_speakers=10
            )

            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment_result = self._transcribe_segment(audio_path, turn, speaker)
                if segment_result:
                    results.append(segment_result)

            return sorted(results, key=lambda x: x['timestamp'][0])

        except Exception as e:
            logger.error("Error in process_audio: %s", str(e))
            raise

    def _transcribe_segment(self, audio_path: str, turn, speaker: str) -> Dict:
        """Transcribe a single audio segment"""
        temp_path = os.path.join(self.temp_dir, f"segment_{turn.start:.2f}.wav")
        try:
            segment = AudioSegment.from_wav(audio_path)[int(turn.start * 1000):int(turn.end * 1000)].normalize()
            segment.export(temp_path, format="wav")

            result = self.asr_pipeline(temp_path)

            return {
                "speaker": speaker,
                "timestamp": (turn.start, turn.end),
                "text": result.strip()
            }

        except Exception as e:
            logger.error("Error transcribing segment: %s", str(e))
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> bool:
        """Convert audio file to WAV format"""
        try:
            AudioSegment.from_file(input_path).export(output_path, format="wav")
            return True
        except Exception as e:
            logger.error("Error converting audio: %s", str(e))
            return False
