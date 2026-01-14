from abc import ABC, abstractmethod
import numpy as np

class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        pass


class TextToSpeech(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        pass


class LLM(ABC):
    @abstractmethod
    def get_response(self, prompt: str, chat_history: list = None) -> str:
        pass

    @abstractmethod
    def convert_to_provider_format(self, chat_history: list) -> list:
        """
        Convert Gemini-format chat history to provider-specific format.
        Gemini format: [{"role": "user"/"model", "parts": [{"text": "..."}]}]
        """
        pass

    @abstractmethod
    def convert_from_provider_format(self, provider_history: list) -> list:
        """
        Convert provider-specific format to Gemini format.
        Returns: [{"role": "user"/"model", "parts": [{"text": "..."}]}]
        """
        pass


class WakeWordDetector(ABC):
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> bool:
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def frame_length(self) -> int:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass
