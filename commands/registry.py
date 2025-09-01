
from commands.definitions.help import HelpCommand
from commands.definitions.clear import ClearCommand
from commands.definitions.login import LoginCommand
from commands.definitions.model import ModelCommand
from commands.definitions.mcp_servers import MCPServers

class CommandRegistry:
    def __init__(self, agent,sessions:dict):
        self.agent = agent
        self._commands = {
            "help": HelpCommand(self.agent, self),
            "clear": ClearCommand(self.agent),
            "login": LoginCommand(self.agent),
            "model": ModelCommand(self.agent),
            "use" : MCPServers(self.agent,sessions)
        }
        self.sessions = sessions


    def get_commands(self):
        return self._commands

    def handle_command(self, command_input: str):
        if not command_input.startswith('/'):
            return None

        parts = command_input[1:].split()
        command_name = parts[0]
        args = parts[1:]

        command = self._commands.get(command_name)
        
        if command_name=="use" and args==[] :
            return list(self.sessions.keys())
        elif command:
            return command.execute(args)
        else:
            return "Unknown command. Type /help for a list of commands."
