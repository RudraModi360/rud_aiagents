import asyncio
import json
from rich import print
from ollama import Client
from typing import List, Dict, Any, Callable, Awaitable
from prompts.system_msg import get_system_prompt
from tools.tool_schemas import ALL_TOOL_SCHEMAS, DANGEROUS_TOOLS, APPROVAL_REQUIRED_TOOLS
from tools.tools import execute_tool


class OllamaAgent:
    def __init__(
        self,
        model: str = 'gpt-oss:20b-cloud',
        temperature: float = 0.7,
        system_message: str = None,
        debug: bool = False
    ):
        self.model = model
        self.temperature = temperature
        self.client = Client()  # Assumes default host, or configure as needed
        self.messages: List[Dict[str, Any]] = []
        self.system_message = system_message or self._build_default_system_message()
        self.messages.append({"role": "system", "content": self.system_message})
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
        self.messages = [msg for msg in self.messages if msg.get('role') == 'system']

    async def chat(self, user_input: str):
        user_message = {"role": "user", "content": user_input}
        self.messages.append(user_message)

        max_iterations = 10
        for _ in range(max_iterations):
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=ALL_TOOL_SCHEMAS,
                options={'temperature': self.temperature}
            )

            message = response['message']
            self.messages.append(message)

            if not message.get('tool_calls'):
                if self.on_final_message:
                    self.on_final_message(message.get('content'))
                return

            if self.debug:
                print("Tool Calling : ", message['tool_calls'])

            tool_messages_to_append = []
            for tool_call in message['tool_calls']:
                tool_name = tool_call['function']['name']
                tool_args = tool_call['function']['arguments']

                if self.on_tool_start:
                    self.on_tool_start(tool_name, tool_args)

                needs_approval = tool_name in DANGEROUS_TOOLS or tool_name in APPROVAL_REQUIRED_TOOLS

                if needs_approval and self.on_tool_approval:
                    approved = await self.on_tool_approval(tool_name, tool_args)
                    if not approved:
                        tool_result = {"success": False, "error": "Tool execution denied by user."}
                    else:
                        tool_result = execute_tool(tool_name, tool_args)
                else:
                    tool_result = execute_tool(tool_name, tool_args)

                if self.on_tool_end:
                    self.on_tool_end(tool_name, tool_result)

                tool_message = {
                    "role": "tool",
                    "tool_call_name": tool_call['function']['name'],
                    "content": json.dumps(tool_result),
                }
                tool_messages_to_append.append(tool_message)

            if tool_messages_to_append:
                self.messages.extend(tool_messages_to_append)

        if self.on_final_message:
            self.on_final_message("Max tool iterations reached. Please try again.")


async def main():
    """A simple chatbot-like example to test the OllamaAgent."""
    agent = OllamaAgent(debug=True)

    def print_tool_start(name, args):
        print(f"[Tool Start] Name: {name}, Args: {args}")

    async def ask_for_approval(name, args):
        try:
            response = input(f"Allow tool '{name}' with args {args}? (y/n): ")
            return response.lower() == 'y'
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
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            await agent.chat(user_input)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())