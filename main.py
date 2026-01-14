import os
from dotenv import load_dotenv

from src.gerald import Gerald
from src.providers.wake_word import PorcupineWakeWord
from src.providers.stt import WhisperSTT, ElevenLabsSTT
from src.providers.tts import ElevenLabsTTS
from src.providers.llm import GeminiLLM

from src.config import (
    WAKE_WORD_KEYWORD_PATHS,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_TTS_MODEL,
    ELEVENLABS_STT_MODEL,
    GEMINI_MODEL
)


def main():
    load_dotenv()

    wake_word = PorcupineWakeWord(
        access_key=os.getenv('PICOVOICE_ACCESS_KEY'),
        keyword_paths=WAKE_WORD_KEYWORD_PATHS
    )

    stt = ElevenLabsSTT(api_key=os.getenv('ELEVENLABS_API_KEY'))
    tts = ElevenLabsTTS(api_key=os.getenv('ELEVENLABS_API_KEY'), voice_id=ELEVENLABS_VOICE_ID, model_id=ELEVENLABS_TTS_MODEL)
    llm = GeminiLLM(api_key=os.getenv('GEMINI_API_KEY'), model=GEMINI_MODEL)

    # Create and run assistant
    assistant = Gerald(wake_word, stt, tts, llm)
    assistant.run()


if __name__ == '__main__':
    main()