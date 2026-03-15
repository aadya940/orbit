from orbit import Agent
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    a1 = Agent(
        llm="gemini-3-pro-preview",
        task=(
            "Read my resume from the Desktop: RESUME.pdf. Then open Chrome, go to LinkedIn, "
            "find an Easy Apply internship that fits my background, and apply by uploading "
            "that same resume (Desktop/RESUME.pdf). Apply to exactly two internships."
        ),
        verbose=True,
    )
    await a1.run()


if __name__ == "__main__":
    asyncio.run(main())
