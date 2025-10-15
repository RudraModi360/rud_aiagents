import os
import subprocess
import re
import ast
import requests
import json
# import docker
from typing import Dict, Any, Set,Optional,List,Union
from groq import Groq
from langgraph.store.memory import InMemoryStore



client_groq = Groq()
# --- Tool-related state ---
global_read_files_tracker: Set[str] = set()
# client = docker.from_env()

class ToolResult(dict):
    def __init__(self, success: bool, content: Any = None, error: str = None):
        super().__init__()
        self['success'] = success
        if content is not None:
            self['content'] = content
        if error is not None:
            self['error'] = error
    
    def to_dict(self):
        return {'success': self['success'], 'content': self.get('content', None),'error': self.get('error', None)}

# --- Validators ---
def validate_read_before_edit(file_path: str) -> bool:
    return os.path.abspath(file_path) in global_read_files_tracker

def get_read_before_edit_error(file_path: str) -> str:
    return f"File must be read before editing. Use read_file tool first: {file_path}"

# --- Tool Implementations ---

def read_file(file_path: str, start_line: int = None, end_line: int = None) -> ToolResult:
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return ToolResult(success=False, error="File not found")
        if not os.path.isfile(abs_path):
            return ToolResult(success=False, error="Path is not a file")
        if os.path.getsize(abs_path) > 50 * 1024 * 1024: # 50MB limit
            return ToolResult(success=False, error="File too large (max 50MB)")

        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        global_read_files_tracker.add(abs_path)
        lines = content.splitlines()

        if start_line is not None:
            start_idx = max(0, start_line - 1)
            end_idx = len(lines) if end_line is None else min(len(lines), end_line)
            if start_idx >= len(lines):
                return ToolResult(success=False, error="Start line exceeds file length")
            
            selected_content = "\n".join(lines[start_idx:end_idx])
            return ToolResult(success=True, content=selected_content)
        else:
            return ToolResult(success=True, content=content)

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to read file: {e}")

def create_file(file_path: str, content: str, file_type: str = 'file', overwrite: bool = False) -> ToolResult:
    try:
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path) and not overwrite:
            return ToolResult(success=False, error="File already exists. Use overwrite=true to replace.")

        if file_type == 'directory':
            os.makedirs(abs_path, exist_ok=True)
            return ToolResult(success=True, content=f"Directory created: {file_path}")
        elif file_type == 'file':
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(success=True, content=f"File created: {file_path}")
        else:
            return ToolResult(success=False, error="Invalid file_type. Must be 'file' or 'directory'.")

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to create file/directory: {e}")

def edit_file(file_path: str, old_text: str, new_text: str, replace_all: bool = False) -> ToolResult:
    if not validate_read_before_edit(file_path):
        return ToolResult(success=False, error=get_read_before_edit_error(file_path))
    try:
        abs_path = os.path.abspath(file_path)
        with open(abs_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        if old_text not in original_content:
            return ToolResult(success=False, error="old_text not found in file.")

        if replace_all:
            updated_content = original_content.replace(old_text, new_text)
        else:
            updated_content = original_content.replace(old_text, new_text, 1)
        
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return ToolResult(success=True, content=f"Successfully edited file: {file_path}")

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to edit file: {e}")

def delete_file(file_path: str, recursive: bool = False) -> ToolResult:
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return ToolResult(success=False, error="Path not found.")

        if os.path.isdir(abs_path):
            if recursive:
                import shutil
                shutil.rmtree(abs_path)
                return ToolResult(success=True, content=f"Recursively deleted directory: {file_path}")
            else:
                if os.listdir(abs_path):
                    return ToolResult(success=False, error="Directory is not empty. Use recursive=true to delete.")
                os.rmdir(abs_path)
                return ToolResult(success=True, content=f"Deleted empty directory: {file_path}")
        else:
            os.remove(abs_path)
            return ToolResult(success=True, content=f"Deleted file: {file_path}")

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to delete: {e}")

def list_files(directory: str = '.', pattern: str = '*.*', recursive: bool = False, show_hidden: bool = False) -> ToolResult:
    try:
        abs_path = os.path.abspath(directory)
        if not os.path.isdir(abs_path):
            return ToolResult(success=False, error="Directory not found.")

        file_list = []
        if recursive:
            for root, dirs, files in os.walk(abs_path):
                if not show_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    files = [f for f in files if not f.startswith('.')]
                
                for file in files:
                    if re.match(pattern.replace('*.*', '.*').replace('*', '.*'), file):
                        file_list.append(os.path.join(root, file))
        else:
            for item in os.listdir(abs_path):
                if not show_hidden and item.startswith('.'):
                    continue
                if re.match(pattern.replace('*.*', '.*').replace('*', '.*'), item):
                    file_list.append(os.path.join(abs_path, item))

        return ToolResult(success=True, content=file_list)

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to list files: {e}")

def web_search(user_input: str) -> ToolResult:
    try:
        response = client_groq.chat.completions.create(
                model="compound-beta",
                messages=[
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            )
        return ToolResult(success=True, content=response.choices[0].message.content)

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to search web: {e}")

def execute_command(command: str, command_type: str, working_directory: str = None, timeout: int = 300) -> ToolResult:
    try:
        if command_type == 'python':
            escaped = command.replace('"', '\\"')
            command = f"python -c '{escaped}' "


        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_directory,
            timeout=timeout
        )

        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            return ToolResult(success=False, content=output, error=f"Command failed with exit code {result.returncode}")
        
        return ToolResult(success=True, content=output)

    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error="Command timed out.")
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to execute command: {e}")

def sandbox_code_execute(code: str, timeout: int = 60) -> ToolResult:
    """
    Executes Python code in a restricted environment and returns the output.
    """
    try:
        process = subprocess.run(
            ['python', '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        output = f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"

        if process.returncode == 0:
            return ToolResult(success=True, content=output)
        else:
            return ToolResult(success=False, content=output, error=f"Code execution failed with exit code {process.returncode}")

    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error="Code execution timed out.")
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to execute code: {e}")
    

# --- Tool Implementations ---

def search_files(pattern: str, file_pattern: str = '*', directory: str = '.', case_sensitive: bool = False,
                 pattern_type: str = 'substring', file_types: Optional[List[str]] = None,
                 exclude_dirs: Optional[List[str]] = None, exclude_files: Optional[List[str]] = None,
                 max_results: int = 100, context_lines: int = 0, group_by_file: bool = False) -> ToolResult:
    """
    Search for pattern in files.
    """
    try:
        import fnmatch
        from pathlib import Path
        import difflib
        results = []
        abs_dir = os.path.abspath(directory)
        if not os.path.isdir(abs_dir):
            return ToolResult(success=False, error="Directory not found.")
        # Prepare regex if needed
        if pattern_type == 'regex':
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        # Walk files
        for root, dirs, files in os.walk(abs_dir):
            # Exclude dirs
            if exclude_dirs:
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                # Exclude files by pattern
                if exclude_files and any(fnmatch.fnmatch(file, pat) for pat in exclude_files):
                    continue
                if not fnmatch.fnmatch(file, file_pattern):
                    continue
                if file_types:
                    ext = os.path.splitext(file)[1].lstrip('.')
                    if ext not in file_types:
                        continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                except Exception:
                    continue
                for idx, line in enumerate(lines, start=1):
                    matched = False
                    match_text = None
                    if pattern_type == 'substring':
                        if (line.find(pattern) != -1) if case_sensitive else (pattern.lower() in line.lower()):
                            matched = True
                            match_text = pattern
                    elif pattern_type == 'exact':
                        if (line.strip() == pattern) if case_sensitive else (line.strip().lower() == pattern.lower()):
                            matched = True
                            match_text = pattern
                    elif pattern_type == 'regex':
                        if regex.search(line):
                            matched = True
                            match_text = regex.search(line).group(0)
                    elif pattern_type == 'fuzzy':
                        # simple fuzzy: ratio > 0.8
                        ratio = difflib.SequenceMatcher(None, pattern, line.strip()).ratio()
                        if ratio > 0.8:
                            matched = True
                            match_text = line.strip()
                    if matched:
                        # context
                        start = max(0, idx - context_lines - 1)
                        end = min(len(lines), idx + context_lines)
                        context = ''.join(lines[start:end]).strip()
                        res = {
                            'file': file_path,
                            'line': idx,
                            'match': match_text,
                            'context': context
                        }
                        results.append(res)
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        if group_by_file:
            grouped = {}
            for r in results:
                grouped.setdefault(r['file'], []).append(r)
            content = grouped
        else:
            content = results
        return ToolResult(success=True, content=content)
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to search files: {e}")

def url_fetch(url: str) -> ToolResult:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        return ToolResult(success=True, content=response.text)
    except requests.exceptions.RequestException as e:
        return ToolResult(success=False, error=f"Failed to fetch URL: {e}")

def fast_grep(keyword: str, directory: str = '.', file_pattern: Optional[str] = None) -> ToolResult:
    """
    Uses ripgrep (rg) to efficiently search for a keyword in a directory.
    """
    try:
        import shutil
        if not shutil.which('rg'):
            return ToolResult(success=False, error="ripgrep (rg) is not installed or not in PATH. This tool requires it for fast searching.")

        abs_path = os.path.abspath(directory)
        if not os.path.isdir(abs_path):
            return ToolResult(success=False, error="Directory not found.")

        command = ['rg', '--json', keyword, abs_path]
        if file_pattern:
            command.extend(['--glob', file_pattern])

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0 and result.returncode != 1: # rg exits 1 if no matches found
            return ToolResult(success=False, error=f"ripgrep command failed with exit code {result.returncode}:\n{result.stderr}")

        # Parse the JSON output
        matches = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    if data['type'] == 'match':
                        matches.append({
                            'file': data['data']['path']['text'],
                            'line_number': data['data']['line_number'],
                            'text': data['data']['lines']['text'].strip()
                        })
                except (json.JSONDecodeError, KeyError):
                    # Ignore lines that are not valid JSON matches (like summaries)
                    pass
        
        if not matches and result.returncode == 1:
             return ToolResult(success=True, content="No matches found.")

        return ToolResult(success=True, content=matches)

    except Exception as e:
        return ToolResult(success=False, error=f"Failed to execute fast_grep: {e}")

# --- Tool Registry and Execution ---

TOOL_REGISTRY = {
    "read_file": read_file,
    "create_file": create_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "list_files": list_files,
    "search_files":search_files,
    "execute_command": execute_command,
    "code_execute": sandbox_code_execute,
    "web_search": web_search,
    "url_fetch": url_fetch,
    "fast_grep": fast_grep
}

def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
    if tool_name not in TOOL_REGISTRY:
        return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
    
    tool_function = TOOL_REGISTRY[tool_name]
    try:
        return tool_function(**tool_args)
    except Exception as e:
        return ToolResult(success=False, error=f"Error executing tool {tool_name}: {e}")