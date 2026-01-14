import sounddevice as sd
from scipy.io.wavfile import write
import os

SAMPLE_RATE = 16000
DURATION = 1.5  # seconds
KEYWORD = "help"
OUT_DIR = f"data/{KEYWORD}"

os.makedirs(OUT_DIR, exist_ok=True)

for i in range(50):
    input(f"Press Enter and say '{KEYWORD}' ({i+1}/50)...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='int16')
    sd.wait()
    filename = f"{OUT_DIR}/{KEYWORD}_{i}.wav"
    write(filename, SAMPLE_RATE, audio)
    print("Saved:", filename)
