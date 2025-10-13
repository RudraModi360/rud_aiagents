# Python Agent Project

## Project Overview

This repository contains a versatile and extensible **Python Agent** framework. Initially designed for simple command-line operations, it has evolved to include a separate backend service, voice capabilities, persistent memory, and a more complex tool management system. The structure is modular, separating command definitions, core agent logic, backend services, tool integrations, and utility settings.


## Project Structure

The project is organized into several key directories and files, each serving a specific purpose in the overall functionality of the Python Agent framework.

```
D:/rud_aiagents/
├── __init__.py             # Marks the root as a Python package
├── main.py                 # Main entry point for the CLI agent
├── Dockerfile              # Docker configuration for containerizing the agent
├── mcp_tools.json          # Tool configuration for the MCP agent
├── memory_test.py          # Script for testing memory functionality
├── check.py                # General-purpose checking script
├── backend/                # Contains a separate backend service
│   ├── Agent.py            # Agent logic specific to the backend
│   ├── main.py             # Entry point for the backend service
│   ├── mcp_tools.json      # Tool configuration for the backend agent
│   └── requirements.txt    # Python dependencies for the backend
├── commands/               # Command infrastructure for the CLI
│   ├── base.py             # Base class for all commands
│   ├── registry.py         # Manages the registration and discovery of commands
│   └── definitions/        # Concrete command implementations (help, login, etc.)
├── core/                   # Core functionality of the agent
│   ├── agent.py            # The central Agent class for the primary CLI
│   ├── agent_mcp.py        # An alternative agent implementation (Multi-Context Prompting)
│   ├── cli.py              # Handles the primary command-line interface
│   ├── cli_mcp.py          # CLI handler for the MCP agent
│   ├── memory.py           # Context management and automatic summarization (NEW)
│   └── narrator.py         # Handles descriptive text generation or voice narration
├── tools/                  # Built-in tool definitions and schemas
│   ├── tools.py            # Collection of core tool functions
│   └── tool_schemas.py     # Pydantic models for tool inputs and outputs
├── utils/                  # Helper utilities
│   ├── fetch_json.py       # Utility for fetching JSON data
│   └── local_settings.py   # Stores local configuration (API keys, defaults)
└── voice/                  # Voice-related functionalities
    ├── check.py            # Checking script for voice components
    └── simple_voice_output.py # Script for simple text-to-speech output
```

---

## Brief Description of Each Component

| Path                      | Purpose                                                                                             |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| `main.py`                 | The main entry point for starting the interactive CLI agent.                                        |
| `mcp_tools.json`          | A JSON file defining the tools available to the `agent_mcp`, allowing for dynamic tool loading.     |
| `backend/`                | A self-contained backend service, likely an API, that runs its own agent instance.                  |
| `commands/`               | Manages the creation, registration, and execution of user-facing commands in the CLI.               |
| `core/agent.py`           | The primary `Agent` class that orchestrates command execution, tool usage, and state management.    |
| `core/agent_mcp.py`       | An alternative `Agent` class, likely for more complex, multi-step reasoning tasks.                  |
| `core/cli.py` / `cli_mcp.py`| Manages the user interaction loop for the respective agent, handling input and displaying output.   |
| `core/memory.py`          | **NEW**: Intelligent context management with automatic summarization, reducing token usage by ~70%. |
| `core/narrator.py`        | Provides descriptive outputs or potentially integrates with voice systems.                          |
| `tools/`                  | Contains the schemas and implementations for the various tools the agent can use (e.g., file I/O).  |
| `utils/`                  | A collection of shared helper functions and configuration loaders.                                  |
| `voice/`                  | Contains scripts for voice output, enabling the agent to speak its responses.                       |

---

## Tools and Extensibility

The agent's capabilities are extended through a robust tool system. Tools are defined in `tools/` and their availability can be configured via `mcp_tools.json`. This allows for different agent instances (e.g., `agent.py` vs. `agent_mcp.py`) to have different sets of capabilities.

### Adding New Tools
1.  **Implement the Tool:** Create the tool's function in `tools/tools.py` or a new module within `tools/`.
2.  **Define the Schema:** Create a Pydantic model in `tools/tool_schemas.py` to define the tool's input parameters.
3.  **Register the Tool:** If using `mcp_tools.json`, add a new entry describing the tool, its purpose, and its parameters.

---

## New Features

### 🚀 Context Management (v1.0)

The agent now includes intelligent context management that automatically reduces token usage by ~70% while maintaining conversation quality:

- **Automatic Summarization**: Old messages are compressed when approaching token limits
- **Token Estimation**: Accurate counting using tiktoken (with heuristic fallback)
- **Recent Context Preservation**: Always keeps recent messages intact for continuity
- **Backwards Compatible**: Existing code works without changes

See [docs/CONTEXT_MANAGEMENT.md](docs/CONTEXT_MANAGEMENT.md) for detailed documentation.

**Quick Start:**
```python
from core.agent_mcp import MCPAgent

# Create agent with context management
agent = MCPAgent(
    max_context_tokens=6000,
    memory_summarization_threshold=0.75
)

# Use normally - context is managed automatically
await agent.chat(sessions, "Your message")

# Check memory stats
stats = agent.memory.get_stats()
print(f"Using {stats['total_tokens']} tokens ({stats['utilization']})")
```

---

## Contributing

Contributions are welcome. Please follow the existing coding style and update this README with any significant changes to the architecture or addition of new top-level modules.