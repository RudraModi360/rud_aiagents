from typing import List, Dict, Any
from groq import Groq

class MemoryManager:
    def __init__(self, client: Groq, model: str = "llama-3.1-8b-instant", window_size: int = 5):
        self.client = client
        self.model = model
        self.window_size = window_size
        self.messages: List[Dict[str, Any]] = []
        self.summary = ""

    def add_message(self, message: Dict[str, Any]):
        self.messages.append(message)
        self.prune()

    def get_context(self) -> List[Dict[str, Any]]:
        context = []
        if self.summary:
            context.append({"role": "system", "content": f"Summary of earlier conversation:\n{self.summary}"})
        context.extend(self.messages)
        return context

    def prune(self):
        if len(self.messages) > self.window_size:
            messages_to_summarize = self.messages[: -self.window_size]
            self.messages = self.messages[-self.window_size:]
            self.summarize(messages_to_summarize)

    def summarize(self, messages: List[Dict[str, Any]]):
        prompt = "Summarize the following conversation in a concise paragraph. Focus on key facts and user preferences:\n\n"
        for msg in messages:
            prompt += f"{msg['role']}: {msg['content']}\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a summarization expert."}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
            )
            new_summary = response.choices[0].message.content
            if self.summary:
                self.summary = f"{self.summary}\n{new_summary}"
            else:
                self.summary = new_summary
        except Exception as e:
            print(f"[MemoryManager] Summarization failed: {e}")

    def clear(self):
        self.messages = []
        self.summary = ""
