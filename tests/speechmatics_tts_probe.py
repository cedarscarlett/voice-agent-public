import asyncio
import os
import sys
import time

from speechmatics.tts import AsyncClient, OutputFormat, Voice


TEXT = "Hello! This is a Speechmatics text to speech test."
VOICE = Voice.SARAH
OUTPUT_FILE = "speechmatics_test.pcm"  # raw PCM16 16kHz mono


async def main() -> None:
    api_key = os.getenv("SPEECHMATICS_API_KEY")
    if not api_key:
        print("ERROR: SPEECHMATICS_API_KEY not set")
        sys.exit(1)

    print("Starting Speechmatics TTS probe")
    print("Text:", repr(TEXT))
    print("Voice:", VOICE)
    print("Output:", OUTPUT_FILE)

    t0 = time.time()

    async with AsyncClient(api_key=api_key) as client:
        print("Calling client.generate()...")

        try:
            # IMPORTANT: await generate FIRST
            response_cm = await asyncio.wait_for(
                client.generate(
                    text=TEXT,
                    voice=VOICE,
                    output_format=OutputFormat.RAW_PCM_16000,
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            print("ERROR: client.generate() timed out")
            return

        print("client.generate() returned, entering response context")

        pcm_bytes = b""
        first_chunk = True

        async with response_cm as response:
            print("Response context entered")

            async for chunk in response.content.iter_chunked(4096):
                if first_chunk:
                    print("First audio chunk received:", len(chunk), "bytes")
                    first_chunk = False
                pcm_bytes += chunk

        dt = time.time() - t0

    print("Synthesis complete")
    print("Total PCM bytes:", len(pcm_bytes))
    print("Estimated duration (ms):", round(len(pcm_bytes) / (2 * 16000) * 1000, 1))
    print("Elapsed wall time (s):", round(dt, 2))

    if len(pcm_bytes) == 0:
        print("ERROR: No audio returned")
        return

    if pcm_bytes[:4] == b"RIFF":
        print("ERROR: Output is WAV, not RAW PCM")
        return

    with open(OUTPUT_FILE, "wb") as f:
        f.write(pcm_bytes)

    print("Wrote raw PCM to", OUTPUT_FILE)
    print("To listen:")
    print("  ffplay -f s16le -ar 16000 -ac 1", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
