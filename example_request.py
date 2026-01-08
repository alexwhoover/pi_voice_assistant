from google import genai
import os
import pvporcupine
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_gemini_response(prompt):
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt
    )

    return response.text

# Porcupine used for wake word detection
porcupine = pvporcupine.create(
    access_key=os.getenv('PICOVOICE_ACCESS_KEY'),
    keyword_paths=['./wake_word.ppn']
)

print("Listening for wake word...")

try:
    # InputStream opens a PortAudio input stream
    # At 512 frames per block and 16000 Hz, each block should contain 32ms of audio
    # These values are required by Porcupine
    with sd.InputStream(samplerate=porcupine.sample_rate, # Sampling frequency in Hz
                        channels=1, # Number of input channels
                        dtype='int16', # Data type (int16 for Porcupine)
                        blocksize=porcupine.frame_length # Number of frames per block
                        ) as stream:
        while True:

            # Read the audio data. stream.read returns [data, overflowed].
            pcm, _ = stream.read(porcupine.frame_length)
            pcm = pcm.flatten()

            # porcupine.process takes a frame of the incoming audio stream and emits the detection result.
            # the porcupine object maintains an internal state since keyword can span multiple blocks
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wake word detected!")

                prompt = "List all the planets in our solar system."
                response = get_gemini_response(prompt)
                print(response)

finally:
    porcupine.delete()

