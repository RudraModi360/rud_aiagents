
import asyncio
import json
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent import MCPAgent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

with open("mcp_tools.json","r") as f:
    mcp_servers=json.loads(f.read())['mcpServers']

sessions = {}
mcp_clients = {}

async def init_sessions():
    for name, args in mcp_servers.items():
        command = args.get('command')
        command_args = args.get('args', [])
        
        # Ensure the command is a list of strings
        if isinstance(command, str):
            full_command = [command] + command_args
        else:
            full_command = command + command_args
            
        params = StdioServerParameters(command=full_command[0], args=full_command[1:])

        client = stdio_client(params)
        mcp_clients[name] = client
        read, write = await client.__aenter__()

        session = ClientSession(read, write)
        await session.__aenter__()

        await session.initialize()

        sessions[name] = session
        print(f"[SYSTEM] Connected to MCP server: {name}")

async def shutdown_sessions():
    for session in sessions.values():
        await session.__aexit__(None, None, None)
    for client in mcp_clients.values():
        await client.__aexit__(None, None, None)

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str

agent = MCPAgent(model="openai/gpt-oss-20b")

@app.on_event("startup")
async def startup_event():
    await init_sessions()
    # The on_final_message callback is not needed anymore as the chat method will return the final message.
    # agent.set_tool_callbacks(
    #     on_final_message=lambda msg: print(f"Final message: {msg}")
    # )

@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_sessions()

@app.post("/chat")
async def chat(request: ChatRequest):
    response = await agent.chat(sessions, request.user_input)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
