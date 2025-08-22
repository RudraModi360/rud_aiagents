
from commands.base import Command
from core.agent import Agent

class ClearCommand(Command):
    def __init__(self, agent: Agent):
        super().__init__("clear", "Clear chat history and context")
        self.agent = agent

    def execute(self, args):
        self.agent.clear_history()
        message = "Chat history and context cleared."
        self.agent.messages.append({
            "role": "system",
            "content": message
        })
        return message
