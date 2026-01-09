import os
from dotenv import load_dotenv

from gerald import Gerald
from providers import PorcupineWakeWord, WhisperSTT, ElevenLabsSTT, ElevenLabsTTS, GeminiLLM


def main():
    load_dotenv()

    wake_word = PorcupineWakeWord(
        access_key=os.getenv('PICOVOICE_ACCESS_KEY'),
        keyword_paths=['./assets/wake_words/gerald.ppn']
    )
    # stt = WhisperSTT(model="tiny")
    stt = ElevenLabsSTT(api_key=os.getenv('ELEVENLABS_API_KEY'))
    tts = ElevenLabsTTS(api_key=os.getenv('ELEVENLABS_API_KEY'), voice_id="NOpBlnGInO9m6vDvFkFC", model_id="eleven_turbo_v2_5")
    llm = GeminiLLM(api_key=os.getenv('GEMINI_API_KEY'), model="gemini-2.5-flash")

    # Create and run assistant
    assistant = Gerald(wake_word, stt, tts, llm)
    assistant.run()


if __name__ == '__main__':
    main()