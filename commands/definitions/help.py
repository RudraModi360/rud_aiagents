
from commands.base import Command
from core.agent import Agent

class HelpCommand(Command):
    def __init__(self, agent: Agent, command_registry):
        super().__init__("help", "Show help and available commands")
        self.agent = agent
        self.command_registry = command_registry

    def execute(self, args):
        commands = self.command_registry.get_commands()
        command_list = "\n".join([f"/{cmd.name} - {cmd.description}" for cmd in commands.values()])
        
        help_text = f"""Available Commands:
{command_list}

Navigation:
- Use arrow keys to navigate chat history
- Type '/' to see available slash commands

Keyboard Shortcuts:
- Ctrl+C - Exit the application
"""
        # In a real CLI, you would print this to the console.
        # For now, we'll add it as a system message.
        self.agent.messages.append({
            "role": "system",
            "content": help_text
        })
        return help_text
