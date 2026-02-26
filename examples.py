"""
claude_webapi — usage examples
"""
import asyncio
import json
from pathlib import Path

from claude_webapi import ClaudeClient, set_log_level
from claude_webapi.constants import Model
from claude_webapi.exceptions import QuotaExceededError, AuthenticationError

SESSION_KEY     = "sk-ant-YOUR_SESSION_KEY"
ORGANIZATION_ID = "YOUR-ORG-UUID"


# ─── 1. Basic single-turn generation ─────────────────────────────────────────

async def example_basic():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        response = await client.generate_content("What is the capital of France?")
        print(response.text)
        # Or simply: print(response)


# ─── 2. Generation with a file attachment ────────────────────────────────────

async def example_with_file():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        response = await client.generate_content(
            "Summarise this document and list the three key takeaways.",
            files=["report.pdf"],
        )
        print(response.text)


# ─── 3. Multi-turn conversation ───────────────────────────────────────────────

async def example_chat():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        chat = client.start_chat()

        r1 = await chat.send_message("My favourite colour is indigo.")
        print("Claude:", r1.text)

        r2 = await chat.send_message("What did I just tell you?")
        print("Claude:", r2.text)   # Should mention indigo


# ─── 4. Resume a previous conversation ───────────────────────────────────────

async def example_resume():
    # First session
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        chat = client.start_chat()
        await chat.send_message("The codeword is WATERMELON.")
        saved_metadata = chat.metadata

    with open("session.json", "w") as f:
        json.dump(saved_metadata, f)

    # Later session (e.g. after restarting the process)
    with open("session.json") as f:
        loaded_metadata = json.load(f)

    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        resumed_chat = client.start_chat(metadata=loaded_metadata)
        r = await resumed_chat.send_message("What was the codeword?")
        print(r.text)   # → WATERMELON


# ─── 5. Streaming output ─────────────────────────────────────────────────────

async def example_streaming():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        print("Claude: ", end="")
        async for chunk in client.generate_content_stream(
            "Write a haiku about the Python programming language."
        ):
            print(chunk.text_delta, end="", flush=True)
        print()


# ─── 6. Streaming in a chat session ──────────────────────────────────────────

async def example_chat_streaming():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        chat = client.start_chat(model=Model.SONNET)
        print("Claude: ", end="")
        async for chunk in chat.send_message_stream(
            "Explain how Python's GIL works in simple terms."
        ):
            print(chunk.text_delta, end="", flush=True)
        print()

        # Follow-up (context preserved)
        r2 = await chat.send_message("Give me a one-line analogy for it.")
        print("Claude:", r2.text)


# ─── 7. Model selection ──────────────────────────────────────────────────────

async def example_model_selection():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        # Fast and cheap
        r_haiku = await client.generate_content(
            "Give me a quick summary of TCP/IP in one sentence.",
            model=Model.HAIKU,
        )
        print(f"Haiku: {r_haiku.text}")

        # Most capable
        r_opus = await client.generate_content(
            "Design a microservices architecture for a ride-sharing app.",
            model=Model.OPUS,
        )
        print(f"Opus: {r_opus.text[:300]}…")


# ─── 8. System prompt ────────────────────────────────────────────────────────

async def example_system_prompt():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        chat = client.start_chat(
            system_prompt=(
                "You are a Socratic tutor. Never give direct answers. "
                "Only ask guiding questions."
            )
        )
        r = await chat.send_message("What is recursion?")
        print(r.text)


# ─── 9. File download from Claude's sandbox ──────────────────────────────────

async def example_file_download():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        chat = client.start_chat()
        await chat.send_message(
            "Write a Python script that prints the first 10 primes "
            "and save it as primes.py"
        )
        local = await client.download_file(chat.cid, "primes.py", dest="./downloads")
        print(f"Downloaded to: {local}")


# ─── 10. Error handling ───────────────────────────────────────────────────────

async def example_error_handling():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        try:
            r = await client.generate_content("Hello!")
            print(r.text)
        except AuthenticationError:
            print("Bad sessionKey — please re-authenticate.")
        except QuotaExceededError:
            print("Message limit hit. Try again later.")


# ─── 11. Delete a conversation ───────────────────────────────────────────────

async def example_delete():
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        chat = client.start_chat()
        await chat.send_message("Temporary chat.")
        print(f"Conversation ID: {chat.cid}")

        await chat.delete()
        print("Conversation deleted.")


# ─── 12. Logging ─────────────────────────────────────────────────────────────

async def example_logging():
    set_log_level("DEBUG")
    async with ClaudeClient(SESSION_KEY, ORGANIZATION_ID) as client:
        r = await client.generate_content("Hello!")
        print(r.text)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run whichever example you like:
    asyncio.run(example_basic())

