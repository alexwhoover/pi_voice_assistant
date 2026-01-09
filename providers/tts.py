import numpy as np
import sounddevice as sd

from gerald import TextToSpeech

class ElevenLabsTTS(TextToSpeech):
    def __init__(self, api_key: str, voice_id: str, model_id: str = "eleven_turbo_v2_5"):
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id

    def speak(self, text: str) -> None:
        try:
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="pcm_24000"
            )

            with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as stream:
                for chunk in audio_generator:
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(audio_data)

        except Exception as e:
            print(f"Error generating speech: {e}")