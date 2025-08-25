# Python Agent Project

## Project Overview

This repository contains a lightweight, extensible **Python Agent** framework designed to run commands, manage tools, and provide a CLI interface. The structure is modular, separating command definitions, core functionality, tool integrations, and utility settings.

---

## Project Structure

The project is organized into several key directories and files, each serving a specific purpose in the overall functionality of the Python Agent framework.

### Directories

1. **commands/**: Contains the infrastructure for command handling.
   - `base.py`: Defines the base class for all commands, providing a common interface.
   - `registry.py`: Manages the registration of commands, making them discoverable.
   - `definitions/`: Houses concrete implementations of commands.

2. **core/**: Hosts the core functionality of the agent.
   - `agent.py`: Implements the central `Agent` class that orchestrates commands and tools.
   - `cli.py`: Handles the command-line interface, parsing user input and displaying results.

3. **tools/**: Includes definitions for built-in tools.
   - `tools.py`: Contains the collection of tool functions.
   - `tool_schemas.py`: Defines Pydantic models for tool inputs and outputs, ensuring type safety.

4. **utils/**: Provides utility functions and settings.
   - `local_settings.py`: Stores local configuration such as API keys and default settings.

### Files

- `main.py`: The entry point for running the agent.
- `__init__.py`: Marks the directory as a Python package.

## Adding New Tools

To add a new tool:
1. Create a new module in `tools/`.
2. Define a Pydantic schema for the tool's input and output.
3. Implement the tool function.
4. Register the tool in `tools/__init__.py`.

## Contributing

Contributions are welcome. Please follow the existing coding style and update the README as necessary.

```
python_agent/
├── __init__.py                     # Package initializer
├── main.py                         # Entry point for running the agent
├── commands/                       # Command infrastructure
│   ├── __init__.py                 # Makes `commands` a package
│   ├── base.py                     # Base class for all commands
│   ├── registry.py                 # Registry that discovers and loads commands
│   └── definitions/                # Concrete command implementations
│       ├── __init__.py             # Makes `definitions` a package
│       ├── clear.py                # Implements the `clear` command (clears screen)
│       ├── help.py                 # Implements the `help` command (shows help)
│       ├── login.py                # Implements the `login` command (auth handling)
│       └── model.py                # Implements the `model` command (model selection)
├── core/                           # Core engine of the agent
│   ├── __init__.py (generated)     # (optional) package init for core
│   ├── agent.py                    # High‑level Agent class orchestrating commands & tools
│   └── cli.py                      # Command‑line interface handling user input
├── tools/                          # Built‑in tool definitions
│   ├── __init__.py (generated)     # (optional) package init for tools
│   ├── tools.py                    # Definitions of built‑in tools (e.g., web search, file ops)
│   └── tool_schemas.py             # Pydantic schemas describing tool input/output
└── utils/                          # Helper utilities
    ├── __init__.py (generated)     # (optional) package init for utils
    └── local_settings.py           # Local configuration (API keys, defaults)
```

> **Note:** `__pycache__` directories are omitted for brevity; they contain compiled byte‑code.

---

## Brief Description of Each File

| File | Purpose |
|------|---------|
| `__init__.py` (root) | Marks the repository as a Python package; can expose top‑level symbols. |
| `main.py` | Small wrapper that creates an `Agent` instance and starts the CLI. |
| `commands/base.py` | Abstract base class (`BaseCommand`) providing a common interface for all commands. |
| `commands/registry.py` | Scans the `commands/definitions` package, registers each concrete command, and makes them discoverable. |
| `commands/definitions/clear.py` | Implements a simple `clear` command to clear the terminal screen. |
| `commands/definitions/help.py` | Implements a `help` command that lists available commands and their usage. |
| `commands/definitions/login.py` | Handles user authentication / login flow for the agent. |
| `commands/definitions/model.py` | Allows switching or configuring the underlying language model. |
| `core/agent.py` | Central `Agent` class – loads commands, manages tool execution, and maintains state. |
| `core/cli.py` | Parses user input from the terminal, maps it to commands, and displays results. |
| `tools/tools.py` | Collection of built‑in tool functions (e.g., web search, file read/write). |
| `tools/tool_schemas.py` | Pydantic models that define the schema for tool inputs and outputs, ensuring type safety. |
| `utils/local_settings.py` | Stores local configuration such as API keys, default model names, and other environment‑specific settings. |

---

## Tools

The Python Agent comes with the following built-in tools:
1. `read_file`: Read file contents with optional line range.
2. `create_file`: Create NEW files or directories.
3. `edit_file`: Modify EXISTING files by exact text replacement.
4. `delete_file`: Remove files or directories.
5. `search_files`: Find text patterns in files across the codebase.
6. `list_files`: Browse directory contents and file structure.
7. `execute_command`: Run shell commands, scripts.
8. `code_execute`: Run python code.
9. `web_search`: Search the web for information.

The framework is designed to let you plug in custom tools with minimal effort.

1. **Create a new tool module** inside the `tools/` package, e.g., `my_tool.py`.
2. **Define a Pydantic schema** (or reuse `tool_schemas.py`) that describes the input parameters and output format.
3. **Implement the tool function** following the signature:
   ```python
   def my_tool(params: MyToolParams) -> ToolResult:
       # Your logic here
   ```
4. **Register the tool** in `tools/__init__.py` (or modify `tools/tools.py`) by adding it to the exported dictionary:
   ```python
   from .my_tool import my_tool, MyToolParams, MyToolResult

   TOOL_REGISTRY = {
       "my_tool": tool_name() ##class for tool
       # existing tools …
   }
   ```


### Tips
- Keep the tool **stateless** or manage state through the `Agent` instance.
- Write unit tests for your tool to ensure it conforms to the declared schema.
- Document the tool in this README under a new **Custom Tools** section.

---

## Contributing

Feel free to submit pull requests for new commands, tools, or improvements. Follow the existing coding style and update the README when adding new top‑level modules.

