from pydantic import BaseModel, Field
from typing import List, Literal, Optional,Dict,Union,Any

class ToolSchema(BaseModel):
    type: str = "function"
    function: dict

# File Operation Tools
class ReadFileParams(BaseModel):
    file_path: str = Field(..., description='Path to file.')
    start_line: Optional[int] = Field(None, description='Starting line number (1-indexed, optional)', ge=1)
    end_line: Optional[int] = Field(None, description='Ending line number (1-indexed, optional)', ge=1)

READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read file contents with optional line range. REQUIRED before edit_file.",
        "parameters": ReadFileParams.model_json_schema()
    }
}

class CreateFileParams(BaseModel):
    file_path: str = Field(..., description='Path for new file/directory.')
    content: str = Field(..., description='File content (use empty string "" for directories)')
    file_type: Literal['file', 'directory'] = Field('file', description='Create file or directory')
    overwrite: bool = Field(False, description='Overwrite existing file')

CREATE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create NEW files or directories. Check if file exists first.",
        "parameters": CreateFileParams.model_json_schema()
    }
}

class EditFileParams(BaseModel):
    file_path: str = Field(..., description='Path to file to edit.')
    old_text: str = Field(..., description='Exact text to replace.')
    new_text: str = Field(..., description='Replacement text.')
    replace_all: bool = Field(False, description='Replace all occurrences.')

EDIT_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Modify EXISTING files by exact text replacement. Always read_file first.",
        "parameters": EditFileParams.model_json_schema()
    }
}

class DeleteFileParams(BaseModel):
    file_path: str = Field(..., description='Path to file/directory to delete.')
    recursive: bool = Field(False, description='Delete directories and their contents.')

DELETE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "Remove files or directories. Use with caution.",
        "parameters": DeleteFileParams.model_json_schema()
    }
}

# Code Execution Tools
class ExecuteCommandParams(BaseModel):
    command: str = Field(..., description='Shell command to execute.')
    command_type: Literal['bash', 'setup', 'run'] = Field(..., description='Command type.')
    working_directory: Optional[str] = Field(None, description='Directory to run command in.')
    timeout: int = Field(300, description='Max execution time in seconds (1-300)', ge=1, le=300)

EXECUTE_COMMAND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Run shell commands, scripts. Only for commands that complete quickly.",
        "parameters": ExecuteCommandParams.model_json_schema()
    }
}

class ExecuteCode(BaseModel):
    code: str = Field(..., description='The Python code to execute.')
    timeout: int = Field(60, description='Max execution time in seconds (1-300)', ge=1, le=300)

EXECUTE_CODE = {
    "type": "function",
    "function": {
        "name": "code_execute",
        "description": "Executes Python code and returns the output. The code should be self-contained.",
        "parameters": ExecuteCode.model_json_schema()
    }
}

# Information Tools
class SearchFilesParams(BaseModel):
    pattern: str = Field(..., description='Text to search for.')
    file_pattern: str = Field('*', description='File pattern filter (e.g., "*.py").')
    directory: str = Field('.', description='Directory to search in.')
    case_sensitive: bool = Field(False, description='Case-sensitive search.')
    pattern_type: Literal['substring', 'regex', 'exact', 'fuzzy'] = Field('substring', description='Match type.')
    file_types: Optional[List[str]] = Field(None, description='File extensions to include.')
    exclude_dirs: Optional[List[str]] = Field(None, description='Directories to skip.')
    exclude_files: Optional[List[str]] = Field(None, description='File patterns to skip.')
    max_results: int = Field(100, description='Maximum results to return (1-1000)', ge=1, le=1000)
    context_lines: int = Field(0, description='Lines of context around matches (0-10)', ge=0, le=10)
    group_by_file: bool = Field(False, description='Group results by filename.')

SEARCH_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Find text patterns in ficles across the codebase.",
        "parameters": SearchFilesParams.model_json_schema()
    }
}

class ListFilesParams(BaseModel):
    directory: str = Field('.', description='Directory path to list.')
    pattern: str = Field('*', description='File pattern filter.')
    recursive: bool = Field(False, description='List subdirectories recursively.')
    show_hidden: bool = Field(False, description='Include hidden files.')

LIST_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "Browse directory contents and file structure.",
        "parameters": ListFilesParams.model_json_schema()
    }
}

class WebSearchParams(BaseModel):
    user_input: str = Field('.', description='content to search for.')

WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Give the actual task to search over the internet.",
        "parameters": WebSearchParams.model_json_schema()
    }
}

class UrlFetchParams(BaseModel):
    url: str = Field(..., description='URL to fetch content from.')

URL_FETCH_SCHEMA = {
    "type": "function",
    
    "function": {
        "name": "url_fetch",
        "description": "Fetch content from a URL.",
        "parameters": UrlFetchParams.model_json_schema()
    }
}

class ManageMemoryParams(BaseModel):
    input_data: Union[str, Dict[str, Any]] = Field(
        ...,
        description="Input for managing memory. Can be a plain string or a structured JSON object."
    )


MANAGE_MEMORY = {
    "type": "function",
    "function": {
        "name": "manage_memory",
        "description": "Store or update information in memory.",
        "parameters": ManageMemoryParams.model_json_schema()
    }
}


# 🔹 Search Memory: accepts only string
class SearchMemoryParams(BaseModel):
    query: str = Field(
        ...,
        description="The search query string to look up in memory."
    )


SEARCH_MEMORY = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": "Search stored information in memory.",
        "parameters": SearchMemoryParams.model_json_schema()
    }
}


class FastGrepParams(BaseModel):
    keyword: str = Field(..., description='The keyword or regex pattern to search for.')
    directory: str = Field('.', description='The directory to search in.')
    file_pattern: Optional[str] = Field(None, description='Glob pattern to filter files to be searched (e.g., "*.py", "**/*.js").')

FAST_GREP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fast_grep",
        "description": "Efficiently search for a keyword or regex pattern across a large number of files in a directory using ripgrep (rg).",
        "parameters": FastGrepParams.model_json_schema()
    }
}


# All tools combined
ALL_TOOL_SCHEMAS = [
    READ_FILE_SCHEMA,
    CREATE_FILE_SCHEMA,
    EDIT_FILE_SCHEMA,
    DELETE_FILE_SCHEMA,
    SEARCH_FILES_SCHEMA,
    LIST_FILES_SCHEMA,
    EXECUTE_COMMAND_SCHEMA,
    EXECUTE_CODE,
    WEB_SEARCH,
    URL_FETCH_SCHEMA,
    MANAGE_MEMORY,
    SEARCH_MEMORY,
    FAST_GREP_SCHEMA
]

# Tool categories
SAFE_TOOLS = ['read_file', 'list_files', 'search_files','sandbox_code_execute', 'url_fetch']
APPROVAL_REQUIRED_TOOLS = ['create_file', 'edit_file']
DANGEROUS_TOOLS = ['delete_file', 'execute_command']