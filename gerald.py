import os
import numpy as np
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import sounddevice as sd

FRAME_LENGTH = 512
SILENCE_THRESHOLD = 10000 # amplitude value for int16 audio
SILENCE_DURATION = 1 # second
HISTORY_LIMIT = 20
INITIAL_PROMPT = "You are a helpful voice assistant named Gerald. You have the persona of a disgruntled cowboy from the American South during the Civil War Era. Keep responses under 100 words, shorter is better. Do not format any text, such as with bold, italics, or lists. Format responses to be easily input into a text-to-speech model. Your location is 26 Park Street, Bristol, United Kingdom."

# ============================================================
# Abstract Base Classes
# ============================================================

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

# ============================================================
# Voice Assistant (Gerald)
# ============================================================

class Gerald:
    def __init__(
        self,
        wake_word: WakeWordDetector,
        stt: SpeechToText,
        tts: TextToSpeech,
        llm: LLM
    ):
        self.wake_word = wake_word
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.chat_history = []
        self._init_chat_history()

    def _init_chat_history(self):
        """
        Initialize chat history with initial guiding prompts to improve answers.
        """
        self.chat_history = [
            {"role": "user", "parts": [{"text": INITIAL_PROMPT}]},
            {"role": "model", "parts": [{"text": "Understood! I'll be helpful, direct and concise."}]}
        ]

    def _is_silent(self, pcm: np.ndarray, silence_threshold: int = SILENCE_THRESHOLD) -> bool:
        """
        Returns true if all discrete audio samples in pcm are below silence threshold
        """
        max_amp = np.abs(pcm).max()
        return max_amp < silence_threshold

    def _record_prompt(self, stream, silence_threshold: int = SILENCE_THRESHOLD, silence_duration: float = SILENCE_DURATION) -> np.ndarray:
        """
        Takes a sounddevice InputStream as input, then records audio until silence is detected.
        Returns a numpy array.
        """
        
        recorded_frames = []
        silence_frames = int(silence_duration * self.wake_word.sample_rate / self.wake_word.frame_length)
        consecutive_silent_frames = 0

        while True:
            pcm, _ = stream.read(self.wake_word.frame_length)
            pcm = pcm.flatten()
            recorded_frames.append(pcm)

            if self._is_silent(pcm, silence_threshold):
                consecutive_silent_frames += 1
            else:
                consecutive_silent_frames = 0

            if consecutive_silent_frames >= silence_frames:
                break

        return np.concatenate(recorded_frames)

    def _handle_prompt(self, text_prompt: str, history_limit: int = HISTORY_LIMIT) -> str:
        response = self.llm.get_response(text_prompt, self.chat_history)

        self.chat_history.append({"role": "user", "parts": [{"text": text_prompt}]})
        self.chat_history.append({"role": "model", "parts": [{"text": response}]})

        if len(self.chat_history) > history_limit:
            # Keep system prompt (first 2 messages) + recent messages
            system_prompt = self.chat_history[:2]
            recent_messages = self.chat_history[-(history_limit - 2):]
            self.chat_history = system_prompt + recent_messages

        return response
    
    def _play_beep(self, frequency=150, duration=0.3):
        """Play a short beep to indicate listening."""
        t = np.linspace(0, duration, int(self.wake_word.sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t) * 0.3
        sd.play(wave.astype(np.float32), samplerate=self.wake_word.sample_rate)
        sd.wait()

    def run(self):
        print("Ready to chat!")

        try:
            with sd.InputStream(
                samplerate=self.wake_word.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.wake_word.frame_length
            ) as stream:
                while True:
                    pcm, _ = stream.read(self.wake_word.frame_length)
                    pcm = pcm.flatten()

                    if self.wake_word.process(pcm):
                        print("Wake word detected.")
                        self._play_beep()

                        audio_prompt = self._record_prompt(stream)
                        text_prompt = self.stt.transcribe(audio_prompt, self.wake_word.sample_rate)

                        if not text_prompt or text_prompt.strip() == "":
                            self.tts.speak("I couldn't hear you.")
                            continue

                        print(f"You said: {text_prompt}\n")
                        response = self._handle_prompt(text_prompt)
                        print(f"{response}\n")

                        self.tts.speak(response)

        finally:
            self.wake_word.cleanup()
