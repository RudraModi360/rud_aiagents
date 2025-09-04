from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
import ast
# Initialize in-memory store with embeddings
store = InMemoryStore(
    index={
        "dims": 784,
        "embed": "ollama:nomic-embed-text:v1.5",  # embedding model
    }
)

# Create tools
memory_tool = create_manage_memory_tool(namespace=("memories",), store=store)
search_tool = create_search_memory_tool(namespace=("memories",), store=store)

# Insert multiple data blocks
data_blocks = [
    "My favorite programming language is Python.",
    "I am currently building an AI chatbot similar to GitHub Copilot.",
    "I love playing chess in my free time.",
    "Snowflake is the cloud platform I am currently learning.",
    "In AUTOSAR, I am studying execution flow of ARXML files.",
    "I enjoy working with embeddings and vector databases.",
    "My contract project involves extracting metadata from PDF files.",
    "I am experimenting with speech-to-text and text-to-speech models.",
    "I want to create persistent memory across chatbot sessions."
]

memory_tool.func({"role": "user", "content": "Hello, my name is Rudra and I live in India."})
for block in data_blocks:
    memory_tool.func(block)

# Run some search queries
queries = [
    "What is my name?",
    "Where do I live?",
    "What is my favorite hobby?",
    "Which programming language do I like?",
    "What project am I working on?",
    "Which platform am I learning?",
    "What file format am I analyzing in AUTOSAR?",
    "What kind of models am I experimenting with?"
]

print(ast.literal_eval(search_tool.func("where rudra lives ?"))[0])