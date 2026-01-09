import numpy as np

from gerald import WakeWordDetector

class PorcupineWakeWord(WakeWordDetector):
    def __init__(self, access_key: str, keyword_paths: list):
        import pvporcupine
        self._porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths
        )

    def process(self, audio_data: np.ndarray) -> bool:
        """
        Returns true if wake word has been detected.
        """
        return self._porcupine.process(audio_data) >= 0

    @property
    def sample_rate(self) -> int:
        return self._porcupine.sample_rate

    @property
    def frame_length(self) -> int:
        return self._porcupine.frame_length

    def cleanup(self) -> None:
        self._porcupine.delete()