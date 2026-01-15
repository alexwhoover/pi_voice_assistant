# Gerald - Voice Assistant

A voice-activated assistant built for Raspberry Pi that uses wake word detection, 
speech-to-text, LLM processing, and text-to-speech to provide a conversational AI experience. Gerald also has the voice and persona of a sage, old wizard.

To begin conversing, simply say **Hey Gerald.**

## Demo
![pi-voice-assistant-image](https://github.com/user-attachments/assets/00090477-58e7-4cf2-8d6a-b7e83ccb0ceb)

https://github.com/user-attachments/assets/5329a0f2-33f6-489a-a46e-006756866f6c


## Installation and Setup Instructions

### Requirements
- Python 3.9+
- Microphone and speaker hardware
- API keys for the following services:
  - [Picovoice](https://picovoice.ai/) (wake word detection)
  - [ElevenLabs](https://elevenlabs.io/) (speech-to-text and text-to-speech)
  - [Google Gemini](https://ai.google.dev/) (LLM)

### Installation
```bash
# Clone the repository
git clone https://github.com/alexwhoover/pi_voice_assistant.git
cd pi_voice_assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root with your API keys:
```env
PICOVOICE_ACCESS_KEY=your_picovoice_key
ELEVENLABS_API_KEY=your_elevenlabs_key
GEMINI_API_KEY=your_gemini_key
```

### Installation of RNNoise Noise Supression Algorithm (Optional)
The RNNoise supression algorithm is a filter for microphone input that improves voice quality and gets rid of background noise such as computer fans. I highly recommend installing this on your system if, like me, you bought the cheapest mic from Amazon with terrible audio quality.

If installing on Raspberry Pi, you must build the LADSPA plugin yourself from the source code, rather than downloading the pre-built linux release. This is because the linux release is made for x86-64 CPUs, while the Raspberry Pi uses an Arm CPU. Simply follow the instructions from the github repository linked below.

[Github Link](https://github.com/werman/noise-suppression-for-voice)

### To Run
```bash
python main.py
```

## Architecture

Gerald follows a modular, provider-based architecture that allows for easy swapping of components:

```
┌─────────────────────────────────────────────────────────────┐
│                         Gerald                              │
│                    (Main Orchestrator)                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────┐
│  Wake Word    │   │   STT / TTS     │   │     LLM       │
│  (Porcupine)  │   │  (ElevenLabs)   │   │   (Gemini)    │
└───────────────┘   └─────────────────┘   └───────────────┘
```

### Core Components

| Component | Interface | Provider(s) |
|-----------|-----------|-------------|
| Wake Word | `WakeWordDetector` | Porcupine |
| Speech-to-Text | `SpeechToText` | ElevenLabs or Whisper |
| Text-to-Speech | `TextToSpeech` | ElevenLabs |
| LLM | `LLM` | Google Gemini |

### Main Loop
```pseudocode
while running:
    listen for audio frames
    if wake_word_detected:
        play_beep()
        record_until_silence()
        text_prompt = transcribe(audio)
        response = llm.get_response(text_prompt, chat_history)
        update_chat_history()
        speak(response)
```

## Configuration

Key settings can be adjusted in `src/config.py`. Of particular importance is the microphone silence threshold, as this threshold will likely depend on your microphone.

## Reflection

### Project Goal
To create a voice assistant where all the components are completely modular. This way if, for example, a new LLM model becomes more popular, it is easy to swap it in. I accomplished this using abstract classes:

```Python
class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_data, sample_rate):
        pass

class TextToSpeech(ABC):
    @abstractmethod
    def speak(self, text):
        pass

class LLM(ABC):
    @abstractmethod
    def get_response(self, prompt, chat_history):
        pass
    ...

class WakeWordDetector(ABC):
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> bool:
        pass
    ...
```

To change to a new provider, simply implement the relevent abstract class and swap it out in main.py.

### Areas for Improvement
1. I would like to test performance on a fully offline implementation using tools like Vost, Whisper, and Ollama. Though, I suspect this would increase the lag between speaking and answering.
2. I would like to improve how the program determines when the user is finished speaking. Currently, it relies on a silence threshold and duration, but a better way would be to use Voice Activity Detection (VAD). VAD identifies whether human speech is present in an audio stream, distinguishing it from silence or background noise. Popular voice assistants like Alexa use this instead.
3. Currently, user speech is recorded fully, then passed to the speech-to-text API. A faster approach would be to begin passing chunks of audio in realtime to the API before the user is finished speaking. Most of the speech-to-text APIs offer a "realtime" endpoint, but I chose not to implement this due to simplicity.

