import audioop
import struct
import sys

FRAME_16K_BYTES = 640  # 20ms PCM16@16k
SAMPLE_WIDTH = 2


class BoundaryProbe:
    def __init__(self, name: str):
        self.name = name
        self.prev_last = None

    def observe(self, pcm: bytes):
        if not pcm:
            return

        n = len(pcm) // 2
        samples = struct.unpack("<" + "h" * n, pcm)

        first = samples[0]
        last = samples[-1]

        if self.prev_last is not None:
            jump = abs(first - self.prev_last)
            if jump > 500:
                print(f"{self.name} LARGE JUMP: {jump}")

        self.prev_last = last


def run_test(pcm_path: str):
    with open(pcm_path, "rb") as f:
        audio = f.read()

    encode_state = None

    probe_16k = BoundaryProbe("PCM16@16k input")
    probe_8k = BoundaryProbe("PCM16@8k after ratecv")

    pos = 0

    while pos + FRAME_16K_BYTES <= len(audio):
        chunk = audio[pos:pos + FRAME_16K_BYTES]
        pos += FRAME_16K_BYTES

        probe_16k.observe(chunk)

        pcm8k, encode_state = audioop.ratecv(
            chunk,
            SAMPLE_WIDTH,
            1,
            16000,
            8000,
            encode_state,
        )

        probe_8k.observe(pcm8k)

        mulaw = audioop.lin2ulaw(pcm8k, SAMPLE_WIDTH)

        if len(mulaw) != 160:
            print("Bad frame length:", len(mulaw))

    print("Done.")


if __name__ == "__main__":
    run_test("tts.pcm")
