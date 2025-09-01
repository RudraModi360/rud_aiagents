from commands.base import Command
from core.agent import Agent

class MCPServers(Command):
    def __init__(self, agent: Agent,sessions:dict):
        super().__init__("use", "Show the List of MCP servers available .")
        self.agent = agent
        self.sessions = sessions
        print("List of MCP Serves available : ",self.sessions.keys())

    def execute(self, args:str):
        print(type(self.sessions[args[0]]))
        return {args[0]:self.sessions[args[0]]}
