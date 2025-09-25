import asyncio
from core.cli_mcp import main


if __name__ == "__main__": 
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")