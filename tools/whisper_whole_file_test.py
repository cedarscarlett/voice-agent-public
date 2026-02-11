# tools/whisper_whole_file_test.py
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np

audio, sr = sf.read("hello.wav", dtype="int16")
assert sr == 16000

audio = audio.astype(np.float32) / 32768.0

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16",
)

segments, _ = model.transcribe(audio, language="en")

for s in segments:
    print(s.text)
