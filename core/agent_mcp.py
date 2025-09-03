import os
import json
from groq import Groq
from typing import List, Dict, Any, Callable, Awaitable
from mcp import ClientSession
from mcp.shared.metadata_utils import get_display_name
from utils.local_settings import ConfigManager
from tools.tool_schemas import ALL_TOOL_SCHEMAS, DANGEROUS_TOOLS, APPROVAL_REQUIRED_TOOLS,SAFE_TOOLS
from tools.tools import execute_tool
from utils.local_settings import ConfigManager


class MCPAgent:
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
        return f"""
    You are "Rudy", a coding assistant powered by {self.model} on Groq and developed by RudraModi360.  
You have access to file-system tools, Groq code execution, shell (Windows PowerShell/CMD), web search, URL fetch, and MCP connectors.  
Your role is to build, modify, and test software in a safe, efficient, and idiomatic way, strictly following the rules below.  

# GLOBAL RULES
1. **Always Check First:**  
   - Files → list_files or read_file  
   - Executables/PATH → execute_command(["where", "<tool>"])  
   - Remote Data → web.search or url_fetch  
2. **Incremental Approach:** Deliver minimal working units first, then extend.  
3. **Prefer Structured APIs:** File ops for files, url_fetch for HTTP, web.search for discovery. No shell hacks unless necessary.  
4. **Safety:** Never run destructive commands (`del /s /q`, `rmdir /s /q`, `rm -rf`) without explicit user approval.  
5. **Filepaths:** Whenever a filepath is required in any tool, **always use the absolute path** instead of relative.  
6. **Output Discipline:** After every tool use, return concise summary (success/failure, brief stdout/stderr, next step).  
7. **Project Conventions:** Always mimic existing style (naming, formatting, architecture). Check imports, config files, or neighbors before introducing new libraries or patterns.  
8. **No Sensitive Data:** Never expose API keys, tokens, or environment variables.  

---

# FILE OPERATIONS (create_file / read_file / edit_file / delete_file / list_files)

- My working dir : {os.getcwd()}

A) **Decision Flow:**  
- New file → list_files → create_file  
- Edit file → read_file → edit_file  
- Replace → read_file → create_file(overwrite=True)  
- Delete → list_files → confirm → delete_file  

B) **Rules:**  
- MUST read_file before editing.  
- NEVER edit without showing old content (header/summary).  
- Every created file must have a header (purpose, inputs, outputs, author/date).  
- Keep files modular and small.  

C) **Output:**  
- Report path, size (bytes/lines), and a 1-line summary after each op.  

---

# SHELL COMMANDS (execute_command) — Windows (PowerShell/CMD)
A) **Use Cases:**  
- Env setup, dependency installs (`pip`, `choco`, `winget`)  
- Build/test commands (`pytest`, `npm run build`)  
- Listing dirs, system checks  

B) **Pre-checks:**  
- execute_command(["cd"])  
- execute_command(["dir"])  
- execute_command(["where", "<tool>"])  
- execute_command(["echo", "%PATH%"])  

C) **Rules:**  
- No file editing via shell — use file ops.  
- Avoid destructive flags. If needed, ask explicitly.  
- Capture stdout + stderr. If fail, show error + fix suggestion.  

D) **Output:**  
- Always show exit code, stdout (trimmed), stderr, next step.  

---

# PYTHON SANDBOX (Groq Execution)
A) **Use Cases:**  
- Algorithm tests, quick calculations, data parsing  
- Validating logic before persisting into project  
- Generating artifacts (plots, CSVs, model outputs)  

B) **Rules:**  
- Always pass user’s request as "user" role to Groq.  
- Imports must be explicit. On failure → show `pip install` fix.  
- Ephemeral code → sandbox-only. Project logic → create_file.  
- Prefer small, testable snippets over huge blocks.  

C) **Execution Pattern:**  
- Wrap in minimal runnable block (imports + print).  
- Extract: reasoning, arguments.code, executed_tools[0].output.  

D) **Output:**  
- Return generated code, runtime output, and test summary (input → output).  
- Generated artifacts → provide file path or create_file.  

---

# WEB (web.search / web.open_url / web.image_query)
A) **Use Cases:**  
- When freshness matters (versions, APIs, news)  
- Unfamiliar term or explicit user request  
- Images when useful (logos, UIs, products)  

B) **Rules:**  
- Verify with web.search if info may have changed in last year.  
- Prefer primary sources (docs, GitHub, vendor sites).  
- Summarize findings; cite URLs properly.  

C) **Output:**  
- Top 3 sources + 1–2 line summary each. Highlight conflicts.  

---

# URL_FETCH
A) **Use Cases:**  
- Fetch raw JSON, CSV, API payloads, downloads  
B) **Rules:**  
- Validate content-type + schema.  
- If large → stream or save via create_file.  
- Never assume structure — inspect first.  
C) **Output:**  
- Show HTTP status, content-type, size, and schema preview.  

---

# MCP (External Connectors)
A) **Use Cases:**  
- Complex AI workflows  
- Calling external services  
B) **Rules:**  
- Validate inputs before sending  
- Split tasks into multiple MCP calls  
- Enforce schema validation on responses  
C) **Output:**  
- Return schema-validated response + confidence + action summary  

---

# ENGINEERING PRACTICES
1. **Tests:** Always add at least one unit test with sample input/output.  
2. **Docs:** Every file → short top-of-file docstring with usage/examples.  
3. **Verification:** Run existing test/lint/build commands. Never assume defaults.  
4. **Explain Critical Commands:** Before destructive/sensitive ops, explain what they do.  
5. **Security:** Never commit or log secrets.  

---

# EXAMPLES
- Safe file creation:  
  if not list_files("C:/Users/Rudra/project/src").contains("main.py"):  
      create_file("C:/Users/Rudra/project/src/main.py", "...")  

- Edit existing file:  
  old = read_file("C:/Users/Rudra/project/src/app.py")  
  new = modify(old)  
  edit_file("C:/Users/Rudra/project/src/app.py", new)  

- Safe shell check:  
  execute_command(["where", "python"])  
  execute_command(["python", "-m", "venv", "C:/Users/Rudra/project/venv"])  

- Python sandbox test:  
  code_execute("import math; print(math.sqrt(101))") → output + summary  
  create_file("C:/Users/Rudra/project/scripts/math_utils.py", "...")  

---

# FINAL MANDATE
If user requests implementation: **DO NOT return text-only.**  
Always use tools (create_file, edit_file, execute_command, code_execute, web.search, url_fetch, MCP).  
Return real outputs, not hypothetical instructions.  

End of directive.  

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
        for msg in self.messages:
            if isinstance(msg,dict):
                if msg['role'] != 'system':
                    self.messages.remove(msg)
            else:
                if msg.role != 'system':
                    self.messages.remove(msg)
        
    def list_mcp_tools_schema(self,tools:list):
        formatted_tools = []
        for tool in tools.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or get_display_name(tool),
                    "parameters": tool.inputSchema,  # already JSON schema
                    
                }
            }
            formatted_tools.append(schema)
        return formatted_tools

    async def chat(self, sessions: Dict[str, ClientSession], user_input: str):
        if not self.client:
            raise ValueError("API key not set. Please set it via set_api_key or GROQ_API_KEY env var.")

        self.messages.append({"role": "user", "content": user_input})
        
        all_mcp_tools_list = []
        mcp_tool_to_session_map = {}
        
        class ToolsContainer:
            def __init__(self, tools):
                self.tools = tools

        for session_name, session in sessions.items():
            mcp_tools = await session.list_tools()
            all_mcp_tools_list.extend(mcp_tools.tools)
            for tool in mcp_tools.tools:
                mcp_tool_to_session_map[tool.name] = session
        
        mcp_tool_names = set(mcp_tool_to_session_map.keys())
        mcp_tools_container = ToolsContainer(all_mcp_tools_list)

        max_iterations = 10
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.list_mcp_tools_schema(mcp_tools_container) + ALL_TOOL_SCHEMAS,
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=self.temperature,
            )

            message = response.choices[0].message
            self.messages.append(message)

            if not message.tool_calls:
                if self.on_final_message:
                    self.on_final_message(message.content)
                return
            
            print("Tool Calling : ", message.tool_calls)
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                is_mcp_tool = tool_name in mcp_tool_names
                tool_args = json.loads(tool_call.function.arguments)

                if self.on_tool_start:
                    self.on_tool_start(tool_name, tool_args)
                
                print("MCP/Built-in : ", is_mcp_tool)
                
                if not is_mcp_tool:  # Built-in tools
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
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    })
                else:  # MCP tools
                    session_for_tool = mcp_tool_to_session_map[tool_name]
                    needs_approval = "create" in tool_name or "delete" in tool_name or "edit" in tool_name or "send" in tool_name
                    
                    tool_result_for_model = None
                    approved = True
                    if needs_approval and self.on_tool_approval:
                        approved = await self.on_tool_approval(tool_name, tool_args)

                    if not approved:
                        error_content = {"success": False, "error": "Tool execution denied by user."}
                        if self.on_tool_end:
                            self.on_tool_end(tool_name, error_content)
                        tool_result_for_model = json.dumps(error_content)
                    else:
                        tool_result = await session_for_tool.call_tool(name=tool_name, arguments=tool_args)
                        print("tool_result : ",tool_result," Type : ",type(tool_result))
                        if tool_result:
                            raw_content = tool_result.content
                            processed_content = raw_content
                            if isinstance(raw_content, list):
                                new_list = []
                                for item in raw_content:
                                    if hasattr(item, 'text') and isinstance(item.text, str):
                                        try:
                                            new_list.append(json.loads(item.text))
                                        except json.JSONDecodeError:
                                            new_list.append(item.text)
                                    else:
                                        new_list.append(item)
                                processed_content = new_list
                            elif hasattr(raw_content, 'text') and isinstance(raw_content.text, str):
                                try:
                                    processed_content = json.loads(raw_content.text)
                                except json.JSONDecodeError:
                                    processed_content = raw_content.text

                            if tool_result.isError:
                                content_for_model = tool_result.structuredContent
                                if content_for_model is None:
                                    content_for_model = {'result': processed_content or 'MCP tool execution failed.'}
                                
                                if self.on_tool_end:
                                    error_for_display = content_for_model
                                    if isinstance(content_for_model, dict):
                                        error_for_display = content_for_model.get('result', json.dumps(content_for_model))
                                    else:
                                        error_for_display = str(content_for_model)
                                    self.on_tool_end(tool_name, {"success": False, "error": error_for_display})
                                
                                tool_result_for_model = json.dumps(content_for_model)
                            else: # success
                                content_for_model = tool_result.structuredContent
                                if content_for_model is None:
                                    content_for_model = processed_content
                                
                                if self.on_tool_end:
                                    self.on_tool_end(tool_name, {"success": True, "content": content_for_model})
                                
                                tool_result_for_model = json.dumps(content_for_model)
                        else: # no result
                            error_content = {"success": False, "error": "MCP tool returned no result."}
                            if self.on_tool_end:
                                self.on_tool_end(tool_name, error_content)
                            tool_result_for_model = json.dumps(error_content)
                
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_for_model,
                    })

        if self.on_final_message:
            self.on_final_message("Max tool iterations reached. Please try again.")