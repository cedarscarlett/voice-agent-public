import asyncio
import os
from openai import AsyncOpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=openai_api_key)

async def get_chat_response(prompt: str):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True,
        service_tier="priority",
    )
    print("LLM CALL END")

    # Process each chunk as it arrives
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

async def main():
    prompt = "Can you hear me?"
    await get_chat_response(prompt)

asyncio.run(main())
