import numpy as np
import scipy.io.wavfile as wav
import tempfile

from gerald import SpeechToText

class WhisperSTT(SpeechToText):
    def __init__(self, model: str = "tiny", device: str = "cpu"):
        import whisper
        self.model = whisper.load_model(model, device=device)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav.write(f.name, sample_rate, audio_data)
            result = self.model.transcribe(f.name, fp16=False)
            return result['text']