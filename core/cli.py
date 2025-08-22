import asyncio
import argparse
from core.agent import Agent
from commands.registry import CommandRegistry

async def tool_approval_callback(tool_name: str, tool_args: dict) -> bool:
    print(f"\n[APPROVAL] Tool requested: {tool_name}")
    print(f"Arguments: {tool_args}")
    response = input("Approve? (y/n): ").lower()
    return response == 'y'

def final_message_callback(message: str):
    print(f"\n[ASSISTANT]\n{message}")

def tool_start_callback(tool_name: str, tool_args: dict):
    print(f"\n[TOOL START] Executing: {tool_name} with args {tool_args}")

def tool_end_callback(tool_name: str, result: dict):
    print(f"\n[TOOL END] {tool_name} finished.")
    if not result['success']:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Result: {result.get('content', 'No content returned')}")

async def main():
    parser = argparse.ArgumentParser(description="A Python agent that uses Groq.")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="The Groq model to use.")
    args = parser.parse_args()

    agent = Agent(model=args.model)
    command_registry = CommandRegistry(agent)

    agent.set_tool_callbacks(
        on_tool_approval=tool_approval_callback,
        on_final_message=final_message_callback,
        on_tool_start=tool_start_callback,
        on_tool_end=tool_end_callback
    )

    print("Welcome to the Python Groq Agent. Type /help for commands.")

    while True:
        try:
            user_input = input("\n[YOU] ")
            if user_input.startswith('/'):
                response = command_registry.handle_command(user_input)
                if response:
                    print(f"\n[SYSTEM] {response}")
            else:
                await agent.chat(user_input)
        except KeyboardInterrupt:
            print("\nExiting agent.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())