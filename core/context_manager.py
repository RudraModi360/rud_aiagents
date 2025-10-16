from __future__ import annotations
import os
from typing import List, Dict, Any

class ContextManager:
    """
    Manages the context for an AI agent based on a multi-layered approach.
    This includes system instructions, domain knowledge, conversation history,
    and specific task requirements.
    """

    def __init__(self, system_prompt: str = None):
        self.system_prompt: str = system_prompt or "You are a helpful assistant."
        self.message_history: List[Dict[str, Any]] = []
        self.output_format: str | None = None
        self.output_format_example: str | None = None
        self.current_task: str | None = None
        self._knowledge_chunks: List[str] = []

    # --- Layer 1: System Context ---
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    # --- Layer 2: Interaction Context (Memory) ---
    def add_message(self, role: str, content: str, tool_calls: List | None = None):
        message = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.message_history.append(message)

    def clear_history(self):
        self.message_history = []

    # --- Layer 3: Response Context ---
    def set_output_format(self, output_format: str | None, example: str | None = None):
        """
        Specifies the desired output format, optionally with a few-shot example.

        Args:
            output_format (str | None): The desired format (e.g., 'JSON', 'Markdown').
            example (str | None): An example of the desired output structure.
        """
        self.output_format = output_format
        self.output_format_example = example

    # --- Layer 4: Task Context ---
    def set_task(self, task: str | None):
        self.current_task = task

    # --- Layer 5: Domain Context (Knowledge) ---
    def load_knowledge_from_file(self, file_path: str, chunk_separator: str = "\n\n"):
        """
        Loads and chunks domain-specific knowledge from a file.
        
        Args:
            file_path (str): Path to the knowledge file.
            chunk_separator (str): The separator used to split the text into chunks (e.g., paragraph).
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self._knowledge_chunks = f.read().split(chunk_separator)
        except Exception:
            self._knowledge_chunks = []

    def _find_relevant_knowledge_snippets(self, query: str, top_k: int = 3) -> str | None:
        """Finds the most relevant knowledge chunks based on simple keyword matching."""
        if not self._knowledge_chunks or not query:
            return None

        query_words = set(query.lower().split())
        scored_chunks = []
        for i, chunk in enumerate(self._knowledge_chunks):
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, i, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        top_chunks = [chunk for score, i, chunk in scored_chunks[:top_k]]
        if not top_chunks:
            return None
        return "\n---\n".join(top_chunks)

    def clear_knowledge(self):
        self._knowledge_chunks = []

    # --- Context Assembly ---
    def assemble_prompt_for_turn(self, user_input: str) -> List[Dict[str, Any]]:
        """Assembles the complete, multi-layered context for the current turn."""
        messages = []
        
        # 1. System Prompt (with integrated Task)
        system_content = self.system_prompt
        if self.current_task:
            system_content += f"\n\nYour primary objective is: {self.current_task}"
        messages.append({"role": "system", "content": system_content})

        # 2. Few-shot example for output format
        if self.output_format and self.output_format_example:
            messages.append({"role": "user", "content": "Show me an example of how to structure the output."})
            messages.append({"role": "assistant", "content": self.output_format_example})

        # 3. Retrieved Knowledge (as a system message)
        relevant_knowledge = self._find_relevant_knowledge_snippets(user_input)
        if relevant_knowledge:
            messages.append({
                "role": "system",
                "content": f"Use the following relevant information from the knowledge base to inform your response:\n---BEGIN KNOWLEDGE---\n{relevant_knowledge}\n---END KNOWLEDGE---"
            })

        # 4. Conversation History
        messages.extend(self.message_history)

        # 5. The actual user input with final instructions
        final_user_message = user_input
        if self.output_format:
             final_user_message += f"\n\n(Remember to provide the final response in {self.output_format} format.)"
        messages.append({"role": "user", "content": final_user_message})

        return messages