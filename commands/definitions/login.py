
from commands.base import Command
from core.agent import Agent

class LoginCommand(Command):
    def __init__(self, agent: Agent):
        super().__init__("login", "Login with your Groq API key")
        self.agent = agent

    def execute(self, args):
        # In a real CLI, this would prompt for the API key.
        # For now, we'll simulate it by taking it as an argument.
        if not args:
            return "Usage: /login <your_api_key>"
        
        api_key = args[0]
        self.agent.set_api_key(api_key)
        message = "API key set successfully."
        self.agent.messages.append({
            "role": "system",
            "content": message
        })
        return message
