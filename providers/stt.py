import numpy as np
import scipy.io.wavfile as wav
import io
import tempfile
from elevenlabs.client import ElevenLabs
import whisper

from gerald import SpeechToText

class WhisperSTT(SpeechToText):
    def __init__(self, model: str = "tiny", device: str = "cpu"):
        self.model = whisper.load_model(model, device=device)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        # Create in-memory WAV file
        buffer = io.BytesIO()
        wav.write(buffer, sample_rate, audio_data)
        buffer.seek(0)
        
        # Whisper requires a file path, so we need to save temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
            f.write(buffer.getvalue())
            f.flush()
            result = self.model.transcribe(f.name, fp16=False)
            return result['text']

class ElevenLabsSTT(SpeechToText):
    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        buffer = io.BytesIO()
        wav.write(buffer, sample_rate, audio_data)
        buffer.seek(0)
        buffer.name = "audio.wav"

        try:
            transcription = self.client.speech_to_text.convert(
                file=buffer,
                model_id="scribe_v2",
                language_code="eng"
            )
            return transcription.text
        except Exception as e:
            print(f"ElevenLabs Transcription Error: {e}")
            return ""