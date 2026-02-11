import wave

with wave.open("hello.wav", "rb") as wf:
    print("sample_rate:", wf.getframerate())
    print("channels:", wf.getnchannels())
    print("sample_width_bytes:", wf.getsampwidth())