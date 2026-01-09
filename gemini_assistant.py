# Basic libraries
import os
import numpy as np
from dotenv import load_dotenv
import time
import scipy.io.wavfile as wav
import tempfile

from google import genai # Google Gemini
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import pvporcupine # Wake Word Detection
import sounddevice as sd # Audio Capture
import whisper # Audio to text transcription

FRAME_LENGTH = 512

def get_gemini_response(client, prompt, chat_history = None):
    contents = chat_history.copy() if chat_history else []
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    try:
        response = client.models.generate_content(
            model = "gemini-2.5-flash-lite",
            contents = contents
        )
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "Rate limit hit. Please try again later."
        else:
            raise e

def text_to_speech(client, text):
    try:
        audio_generator = client.text_to_speech.convert(
            text = text,
            voice_id = "YXpFCvM1S3JbWEJhoskW",
            model_id = "eleven_turbo_v2_5",
            output_format="pcm_24000"
        )

        with sd.OutputStream(samplerate = 24000, channels = 1, dtype = 'int16') as stream:
            for chunk in audio_generator:
                audio_data = np.frombuffer(chunk, dtype = np.int16)
                stream.write(audio_data)

    except Exception as e:
        print(f"Error generating speech: {e}")

def init_porcupine():
    return pvporcupine.create(access_key=os.getenv('PICOVOICE_ACCESS_KEY'), keyword_paths=['./wake_word.ppn'])

def is_silent(pcm, silence_threshold):
    return (np.abs(pcm).max() < silence_threshold)

def record_prompt(stream, sample_rate, silence_threshold = 500, silence_duration = 1.0):
    """
    Record audio until silence is detected for silence_duration seconds.
    """

    recorded_frames = []
    silence_frames = int(silence_duration * sample_rate / FRAME_LENGTH)
    consecutive_silent_frames = 0

    while True:
        pcm, _ = stream.read(FRAME_LENGTH)
        pcm = pcm.flatten()
        recorded_frames.append(pcm)

        if(is_silent(pcm, silence_threshold)):
            consecutive_silent_frames += 1
        else:
            consecutive_silent_frames = 0
        
        if consecutive_silent_frames >= silence_frames:
            break
    
    return np.concatenate(recorded_frames)

def transcribe_audio(model, audio_data, sample_rate):
    # Convert audio_data in numpy array format to a temporary WAV file, for use with OpenAI's Whisper
    with tempfile.NamedTemporaryFile(suffix = '.wav', delete = False) as f:
        wav.write(f.name, sample_rate, audio_data)
        result = model.transcribe(f.name, fp16=False)
        return result['text']

def handle_prompt(gemini_client, text_prompt, chat_history, history_limit = 20):
    if not chat_history:
        chat_history.append({
            "role": "user",
            "parts": [{"text": "You are a helpful voice assistant. Answer questions in a conversational way, keeping responses under 100 words."}]
        })
        chat_history.append({
            "role": "model",
            "parts": [{"text": "Understood! I'll be a helpful voice assistant and keep my responses concise."}]
        })

    response = get_gemini_response(gemini_client, text_prompt, chat_history)

    # Update chat history
    chat_history.append({
        "role": "user",
        "parts": [{"text": text_prompt}]
    })
    chat_history.append({
        "role": "model",
        "parts": [{"text": response}]
    })

    # Limit history length to prevent token overflow
    if len(chat_history) > history_limit:  # Keep last 10 exchanges
        chat_history = chat_history[-history_limit:]

    return response


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Porcupine used for wake word detection
    porcupine = init_porcupine()

    # Load Whisper model
    whisper_model = whisper.load_model("tiny", device = "cpu") # Options: tiny, base, small, medium, large
    gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    eleven_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

    # Initialize chat history
    chat_history = []

    print("Ready to chat!")

    try:
        # InputStream opens a PortAudio input stream
        with sd.InputStream(samplerate=porcupine.sample_rate, # Sampling frequency in Hz
                            channels=1, # Number of input channels
                            dtype='int16', # Data type (int16 for Porcupine)
                            blocksize=porcupine.frame_length # Number of frames per block
                            ) as stream:
            while True:
                # Read the audio data. stream.read returns [data, overflowed].
                pcm, _ = stream.read(porcupine.frame_length)
                pcm = pcm.flatten()
                keyword_index = porcupine.process(pcm)

                # If keyword has been detected
                if keyword_index >= 0:
                    print("Wake word detected.")
                    # Record prompt
                    audio_prompt = record_prompt(stream, porcupine.sample_rate)
                    text_prompt = transcribe_audio(whisper_model, audio_prompt, porcupine.sample_rate)
                    
                    if not text_prompt or text_prompt.strip() == "":
                        print("I couldn't hear you.")
                        continue

                    print(f"You said: {text_prompt} \n")
                    response = handle_prompt(gemini_client, text_prompt, chat_history)
                    print(response + "\n")

                    # Speak back
                    text_to_speech(eleven_client, response)

    finally:
        porcupine.delete()

if __name__ == '__main__':
    main()

