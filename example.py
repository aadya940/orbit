from orbit import Agent
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    a1 = Agent(
        llm="gemini-3-pro-preview",
        task="Open Chrome and navigate to Wikipedia",
        verbose=False,
    )
    await a1.run()


if __name__ == "__main__":
    asyncio.run(main())
