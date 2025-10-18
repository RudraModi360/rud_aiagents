import asyncio
import json
import sys
import os
from rich import print
from ollama import Client
from typing import List, Dict, Any, Callable, Awaitable
from prompts.system_msg import get_system_prompt
from tools.tool_schemas import (
    ALL_TOOL_SCHEMAS,
    DANGEROUS_TOOLS,
    APPROVAL_REQUIRED_TOOLS,
)
from tools.tools import execute_tool
from core.context_manager import ContextManager
from ollama._types import ChatResponse

class OllamaAgent:
    def __init__(
        self,
        model: str = "gpt-oss:20b-cloud",
        temperature: float = 0.2,
        system_message: str = None,
        debug: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.client = Client()  # Assumes default host, or configure as needed

        # Modern context management
        self.system_message = system_message or self._build_default_system_message()
        self.context_manager = ContextManager(system_prompt=self.system_message)

        self.debug = debug
        # Callbacks
        self.on_tool_start: Callable[[str, Dict], None] = None
        self.on_tool_end: Callable[[str, Any], None] = None
        self.on_tool_approval: Callable[[str, Dict], Awaitable[bool]] = None
        self.on_final_message: Callable[[str], None] = None

    def _build_default_system_message(self) -> str:
        return get_system_prompt(self.model)

    def set_tool_callbacks(
        self,
        on_tool_start: Callable[[str, Dict], None] = None,
        on_tool_end: Callable[[str, Any], None] = None,
        on_tool_approval: Callable[[str, Dict], Awaitable[bool]] = None,
        on_final_message: Callable[[str], None] = None,
    ):
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_approval = on_tool_approval
        self.on_final_message = on_final_message

    def clear_history(self):
        self.context_manager.clear_history()

    async def chat(self, user_input: str):
        # Assemble the full context for this turn
        messages = self.context_manager.assemble_prompt_for_turn(user_input)
        # The user input is now part of the assembled messages, so we add it to history here for the next turn
        messages=[]
        messages = [
            {
                "role": "user",
                "content": user_input
            }
        ]
        self.context_manager.add_message("user", user_input)
        max_iterations = 10
        for i in range(max_iterations):
            if self.debug:
                print("[Debug] Messages sent to model:", messages)
            response: ChatResponse = self.client.chat(
            model=self.model, messages=messages, tools=ALL_TOOL_SCHEMAS
                )
            # print("response : ", response)
            message = response.message
            messages.append(message)
            # self.context_manager.add_message(
            #     message["role"], message["content"], message.get("tool_calls")
            # )

            if not message.tool_calls:
                if self.on_final_message:
                    self.on_final_message(message.content)
                # Clear per-turn context like task and output format
                # self.context_manager.set_task(None)
                # self.context_manager.set_output_format(None)
                return

            if self.debug:
                print("Tool Calling : ", message.tool_calls)

            tool_messages_to_append = []
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                if self.on_tool_start:
                    self.on_tool_start(tool_name, tool_args)

                needs_approval = (
                    tool_name in DANGEROUS_TOOLS or tool_name in APPROVAL_REQUIRED_TOOLS
                )

                if needs_approval and self.on_tool_approval:
                    approved = await self.on_tool_approval(tool_name, tool_args)
                    if not approved:
                        tool_result = {
                            "success": False,
                            "error": "Tool execution denied by user.",
                        }
                    else:
                        tool_result = execute_tool(tool_name, tool_args)
                else:
                    tool_result = execute_tool(tool_name, tool_args)

                if self.on_tool_end:
                    self.on_tool_end(tool_name, tool_result)

                tool_message = {
                    "role": "tool",
                    "content": str(tool_result.get("content")),
                    "tool_name": tool_call.function.name,
                }
                # messages.append(tool_message)
                tool_messages_to_append.append(tool_message)

                # Also add the raw tool result to the context manager's history for the next iteration
                self.context_manager.add_message(
                    "tool", str(tool_result.get("content")), [tool_call]
                )

            # The tool results need to be in the message list for the next model call in the loop
            messages.extend(tool_messages_to_append)

        if self.on_final_message:
            self.on_final_message("Max tool iterations reached. Please try again.")


async def main():
    """A simple chatbot-like example to test the OllamaAgent with ContextManager."""
    agent = OllamaAgent(debug=False)  # Set to True for verbose context logging

    # Create a dummy knowledge file for demonstration
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
    with open("knowledge_base/project_info.txt", "w") as f:
        f.write(
            "The project is named 'rud_aiagents'. It is an agentic AI framework. The main agent is OllamaAgent. The project structure includes core, tools, and prompts directories."
        )

    def print_tool_start(name, args):
        print(f"[Tool Start] Name: {name}, Args: {args}")

    async def ask_for_approval(name, args):
        try:
            response = input(f"Allow tool '{name}' with args {args}? (y/n): ")
            return response.lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False

    def print_tool_end(name, result):
        print(f"[Tool End] Name: {name}, Result: {result}")

    def print_final_message(content):
        print(f"[Final Message] {content}")

    agent.set_tool_callbacks(
        on_tool_start=print_tool_start,
        on_tool_end=print_tool_end,
        on_tool_approval=ask_for_approval,
        on_final_message=print_final_message,
    )

    print("Ollama Chatbot. Type 'exit' to quit.")
    print(
        "Commands: /task [description], /knowledge [filepath], /output [format], /clear"
    )

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            if user_input.startswith("/task "):
                agent.context_manager.set_task(user_input.split(" ", 1)[1])
                print(f"[Context] Task set to: {agent.context_manager.current_task}")
                continue
            elif user_input.startswith("/knowledge "):
                agent.context_manager.load_knowledge_from_file(
                    user_input.split(" ", 1)[1]
                )
                print(f"[Context] Knowledge loaded from: {user_input.split(' ', 1)[1]}")
                continue
            elif user_input.startswith("/output "):
                agent.context_manager.set_output_format(user_input.split(" ", 1)[1])
                print(
                    f"[Context] Output format set to: {agent.context_manager.output_format}"
                )
                continue
            elif user_input == "/clear":
                agent.clear_history()
                agent.context_manager.clear_knowledge()
                agent.context_manager.set_task(None)
                agent.context_manager.set_output_format(None)
                print("[Context] All context cleared.")
                continue

            await agent.chat(user_input)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
