import asyncio
from typing import Callable, Any

class Narrator:
    def __init__(self, client=None, model="openai/gpt-oss-20b", on_status: Callable[[str], None] = None):
        self.client = client
        self.model = model
        self.on_status = on_status

    def emit(self, text: str):
        if self.on_status:
            self.on_status(text)

    def say(self, text: str, use_llm: bool = False):
        """Optionally rewrite narration with LLM"""
        if use_llm and self.client:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Rewrite this status update into a short, natural, human-like narration (max 1 sentence)."},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=30,
                    temperature=0.7,
                )
                pretty_text = resp.choices[0].message.content
                if pretty_text and pretty_text.strip():
                    self.emit(pretty_text.strip())
                    return
            except Exception as e:
                self.emit(f"(narration error: {e}) → {text}")
                return
        # fallback if LLM fails or disabled
        self.emit(text)

    async def with_status_updates(self, func: Callable[..., Any], hints=None, delay=3, *args, **kwargs):
        hints = hints or ["⌛ Still working...", "🔎 Processing results...", "⚡ Almost done..."]

        async def ticker():
            for msg in hints:
                await asyncio.sleep(delay)
                self.say(msg, use_llm=True)

        task = asyncio.create_task(ticker())
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        finally:
            task.cancel()
