import os
import json
from groq import Groq
from typing import List, Dict, Any, Callable, Awaitable

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

        # Callbacks
        self.on_tool_start: Callable[[str, Dict], None] = None
        self.on_tool_end: Callable[[str, Any], None] = None
        self.on_tool_approval: Callable[[str, Dict], Awaitable[bool]] = None
        self.on_final_message: Callable[[str], None] = None

    def _build_default_system_message(self) -> str:
        return f"""You are a coding assistant powered by {self.model} on Groq. Tools are available to you. Use tools to complete tasks.

You have access to read-search across my local file-system's file too. here is my user's path  : {os.path.expanduser('~')}
 and my current dir {os.getcwd()}

CRITICAL: For ANY implementation request (building apps, creating components, writing code), you MUST use tools to create actual files. NEVER provide text-only responses for coding tasks that require implementation.

Use tools to:
- Read and understand files (read_file, list_files)
- Create, edit, and manage files (create_file, edit_file, delete_file)
- Execute bash commands (execute_command)
- Run python code in sandbox using the (code_execute)

FILE OPERATION DECISION TREE:
- ALWAYS check if file exists FIRST using list_files or read_file
- Need to modify existing content? → read_file first, then edit_file (never create_file)
- Need to create something new? → list_files to check existence first, then create_file
- File exists but want to replace completely? → create_file with overwrite=True
- Unsure if file exists? → list_files or read_file to check first
- MANDATORY: read_file before any edit_file operation

IMPORTANT: When creating files, keep them focused and reasonably sized. For large applications:
1. Start with a simple, minimal version first
2. Create separate files for different components
3. Build incrementally rather than generating massive files at once

Be direct and efficient.
"""

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