import os
import json
from groq import Groq
from typing import List, Dict, Any, Callable, Awaitable
from prompts.system_msg import get_system_prompt
from tools.tool_schemas import ALL_TOOL_SCHEMAS, DANGEROUS_TOOLS, APPROVAL_REQUIRED_TOOLS
from tools.tools import execute_tool
from utils.local_settings import ConfigManager

class Agent:
    def __init__(
        self,
        model: str = 'openai/gpt-oss-120b',
        temperature: float = 0.7,
        system_message: str = None,
        debug: bool = False
    ):
        self.model = model
        self.temperature = temperature
        self.config_manager = ConfigManager()
        self.api_key = self.config_manager.get_api_key() or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.messages: List[Dict[str, Any]] = []
        self.system_message = system_message or self._build_default_system_message()
        self.messages.append({"role": "system", "content": self.system_message})
        self.debug = debug
        self.max_context_tokens = 8000  # Max tokens for context window

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

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)
        self.config_manager.set_api_key(api_key)

    def clear_history(self):
        self.messages = [msg for msg in self.messages if msg['role'] == 'system']

    async def chat(self, user_input: str):
        if not self.client:
            raise ValueError("API key not set. Please set it via set_api_key or GROQ_API_KEY env var.")

        self.messages.append({"role": "user", "content": user_input})

        max_iterations = 30
        for _ in range(max_iterations):
            self.messages = self.messages
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=ALL_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=self.temperature,
            )


            message = response.choices[0].message
            # print("Message : ",message)
            self.messages.append(message)

            if not message.tool_calls:
                if self.on_final_message:
                    self.on_final_message(message.content)
                return

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if self.on_tool_start:
                    self.on_tool_start(tool_name, tool_args)

                # Approval
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

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    }
                )

        if self.on_final_message:
            self.on_final_message("Max tool iterations reached. Please try again.")