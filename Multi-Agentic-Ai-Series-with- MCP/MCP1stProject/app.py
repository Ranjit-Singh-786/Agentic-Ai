import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
# Set before importing mcp_use so telemetry initializes correctly.
os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"

from mcp_use import MCPAgent, MCPClient
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY missing. Add it in your .env file.")


async def main():
    llm = ChatOpenAI(model="gpt-4o-mini")

    # llm = ChatGroq(model="llama-3.3-70b-versatile")

    config_json_path = "browser_mcp.json" 

    client = MCPClient.from_config_file(config_json_path)
    agent = MCPAgent(client=client, llm=llm,max_steps=15,memory_enabled=True)
    while True:   
        user_input = input("Enter your question: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            agent.clear_conversation_history()
            continue 

        result = await agent.run(user_input)
        print(result)
        print("==============================================================="*2)
        print("\n"*3)
        
    if client and client.sessions:
        await client.close_all_sessions()



if __name__ == "__main__":
    asyncio.run(main())