import numpy as np
import sounddevice as sd
from src.config import SILENCE_THRESHOLD, SILENCE_DURATION, BEEP_FREQUENCY, BEEP_DURATION

class AudioManager:
    def __init__(self, sample_rate: int, frame_length: int):
        self.sample_rate = sample_rate
        self.frame_length = frame_length

    def is_silent(self, pcm: np.ndarray, silence_threshold: int = SILENCE_THRESHOLD) -> bool:
        """
        Returns true if all discrete audio samples in pcm are below silence threshold
        """
        max_amp = np.abs(pcm).max()
        return max_amp < silence_threshold

    def record_until_silence(self, stream, silence_threshold: int = SILENCE_THRESHOLD, silence_duration: float = SILENCE_DURATION) -> np.ndarray:
        """
        Takes a sounddevice InputStream as input, then records audio until silence is detected.
        Returns a numpy array.
        """
        
        recorded_frames = []
        silence_frames = int(silence_duration * self.sample_rate / self.frame_length)
        consecutive_silent_frames = 0

        while True:
            try:
                pcm, overflowed = stream.read(self.frame_length)

                if overflowed:
                    print("Audio buffer overflow detected")

                pcm = pcm.flatten()
                recorded_frames.append(pcm)

                if self.is_silent(pcm, silence_threshold):
                    consecutive_silent_frames += 1

                    # Loop exit condition
                    if consecutive_silent_frames >= silence_frames:
                        break
                else:
                    consecutive_silent_frames = 0
            
            except sd.PortAudioError as e:
                print(f"Audio recording error: {e}")
                break

        return np.concatenate(recorded_frames) if recorded_frames else np.array([], dtype=np.int16)

    def play_beep(self, frequency: int = BEEP_FREQUENCY, duration: float = BEEP_DURATION) -> None:
        """Play a short beep to indicate listening."""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            wave = np.sin(2 * np.pi * frequency * t) * 0.3
            sd.play(wave, self.sample_rate)
            sd.wait()
        except sd.PortAudioError as e:
            print(f"Audio playback error: {e}")

    def play_audio(self, audio_data: np.ndarray) -> None:
        """Play audio data through the default output device."""
        try:
            sd.play(audio_data, self.sample_rate)
            sd.wait()
        except sd.PortAudioError as e:
            print(f"Audio playback error: {e}")