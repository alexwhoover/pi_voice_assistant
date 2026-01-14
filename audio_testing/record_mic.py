import sounddevice as sd
import soundfile as sf
import numpy as np

# Configuration
# Using 48000Hz because that is the native rate of PipeWire/Pulse
FS = 48000  
DURATION = 5  # Seconds
FILENAME = "test.wav"

def record_mic():
    print(f"Recording for {DURATION} seconds...")
    print("Speak into your USB PnP Sound Device now!")
    
    # sd.rec is non-blocking, so it starts in the background
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
    
    sd.wait()
    print("Recording finished.")

    sf.write(FILENAME, recording, FS)
    print(f"File saved as: {FILENAME}")

if __name__ == "__main__":
    try:
        record_mic()
    except Exception as e:
        print(f"An error occurred: {e}")
