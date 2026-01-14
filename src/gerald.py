import numpy as np
from dotenv import load_dotenv
import sounddevice as sd

# Abstract classes
from src.interfaces import WakeWordDetector, SpeechToText, TextToSpeech, LLM

# Audio class
from src.audio_utils import AudioManager

# Configuration
from src.config import INITIAL_PROMPT, HISTORY_LIMIT, STARTUP_MESSAGE

class Gerald:
    """
    Note: I have chosen Gerald's chat history standard to be the same as Google Gemini's API standard.
    If using other LLM, you must define convert_to_provider_format / convert_from_provider_format for chat_history format conversion.
    """
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
        self.audio_manager = AudioManager(
            sample_rate = wake_word.sample_rate,
            frame_length = wake_word.frame_length
        )
        self._init_chat_history()

    def _init_chat_history(self):
        """
        Initialize chat history with initial guiding prompts to improve answers.
        """

        self.chat_history = [
            {"role": "user", "parts": [{"text": INITIAL_PROMPT}]},
            {"role": "model", "parts": [{"text": "Understood! I'll be helpful, direct and concise."}]}
        ]

    def _update_chat_history(self, text_prompt: str, response:str, history_limit: int = HISTORY_LIMIT):
        self.chat_history.append({"role": "user", "parts": [{"text": text_prompt}]})
        self.chat_history.append({"role": "model", "parts": [{"text": response}]})

        if len(self.chat_history) > history_limit:
            # Keep system prompt (first 2 messages) + recent messages
            system_prompt = self.chat_history[:2]
            recent_messages = self.chat_history[-(history_limit - 2):]
            self.chat_history = system_prompt + recent_messages

    def run(self):
        print(STARTUP_MESSAGE)
        self.tts.speak(STARTUP_MESSAGE)

        try:
            with sd.InputStream(
                samplerate=self.wake_word.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.wake_word.frame_length
            ) as stream:
                while True:
                    # MAIN LOOP
                    try:
                        pcm, _ = stream.read(self.wake_word.frame_length)
                        pcm = pcm.flatten()

                        if self.wake_word.process(pcm):
                            print("Wake word detected.")
                            self.audio_manager.play_beep()

                            print("Listening for prompt.")
                            audio_prompt = self.audio_manager.record_until_silence(stream)

                            print("Transcribing.")
                            text_prompt = self.stt.transcribe(audio_prompt, self.wake_word.sample_rate)

                            if text_prompt:
                                response = self.llm.get_response(text_prompt, self.chat_history)
                                self._update_chat_history(text_prompt, response, HISTORY_LIMIT)
                                print(f"User: {text_prompt}")
                                print(f"Gerald: {response}")
                                self.tts.speak(response)
                            else:
                                print("No prompt detected")
                    except sd.PortAudioError as e:
                        print(f"Audio stream error: {e}")
                        break

        except KeyboardInterrupt:
            print("\nGerald shutting down.")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
            self.wake_word.cleanup()
            print("Cleanup complete.")
