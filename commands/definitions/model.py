
from commands.base import Command
from core.agent import Agent

class ModelCommand(Command):
    def __init__(self, agent: Agent):
        super().__init__("model", "Select your Groq model")
        self.agent = agent

    def execute(self, args):
        # In a real CLI, this would show a model selector.
        # For now, we'll take the model name as an argument.
        if not args:
            return f"Current model: {self.agent.model}. Usage: /model <model_name>"
        
        model_name = args[0]
        self.agent.model = model_name
        self.agent.config_manager.set_default_model(model_name)
        message = f"Model set to {model_name}."
        self.agent.messages.append({
            "role": "system",
            "content": message
        })
        return message
