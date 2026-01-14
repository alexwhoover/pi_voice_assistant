# Audio Settings
FRAME_LENGTH = 512
SILENCE_THRESHOLD = 500 # amplitude value for int16 audio
SILENCE_DURATION = 1 # second

# Chat Settings
HISTORY_LIMIT = 20
INITIAL_PROMPT = "You are a helpful voice assistant named Gerald. You have the persona of a disgruntled cowboy from the American South during the Civil War Era. Keep responses under 100 words, shorter is better. Do not format any text, such as with bold, italics, or lists. Format responses to be easily input into a text-to-speech model. Your location is 26 Park Street, Bristol, United Kingdom."

# Wake Word Settings
WAKE_WORD_KEYWORD_PATHS = ['./assets/wake_words/gerald.ppn']

# STT / TSS Settings
## Whisper
WHISPER_MODEL = "tiny"
WHISPER_DEVICE = "cpu"

## Eleven Labs
ELEVENLABS_VOICE_ID = "NOpBlnGInO9m6vDvFkFC"
ELEVENLABS_TTS_MODEL = "eleven_turbo_v2_5"
ELEVENLABS_STT_MODEL = "scribe_v2"
ELEVENLABS_STT_LANGUAGE = "eng"
ELEVENLABS_TTS_SAMPLE_RATE = 24000
ELEVENLABS_TTS_OUTPUT_FORMAT = "pcm_24000"

# LLM Settings
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Audio feedback settings
BEEP_FREQUENCY = 150
BEEP_DURATION = 0.3

# Startup message
STARTUP_MESSAGE = "Gerald is online."