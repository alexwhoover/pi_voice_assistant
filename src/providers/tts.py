import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs

from src.interfaces import TextToSpeech
from src.config import ELEVENLABS_TTS_SAMPLE_RATE, ELEVENLABS_TTS_OUTPUT_FORMAT

class ElevenLabsTTS(TextToSpeech):
    def __init__(self, api_key: str, voice_id: str, model_id: str = "eleven_turbo_v2_5"):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id

    def speak(self, text: str) -> None:
        try:
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=ELEVENLABS_TTS_OUTPUT_FORMAT
            )

            with sd.OutputStream(samplerate=ELEVENLABS_TTS_SAMPLE_RATE, channels=1, dtype='int16') as stream:
                for chunk in audio_generator:
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(audio_data)
        except sd.PortAudioError as e:
            print(f"Audio output error: {e}")
        except Exception as e:
            print(f"Error generating speech: {e}")