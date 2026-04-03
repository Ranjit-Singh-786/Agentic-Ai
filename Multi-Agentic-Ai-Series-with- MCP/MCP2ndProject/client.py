from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

import asyncio
import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent

MATH_SYSTEM_PROMPT = """You can use calculator tools to answer math questions.
Tool arguments must be plain integers only — never nested objects or "function" payloads inside parameters.
For (a + b) × c, prefer add_then_multiply when it applies."""

def _stdio_server(script_name: str) -> dict:
    return {
        "command": sys.executable,
        "args": [str(SCRIPT_DIR / script_name)],
        "transport": "stdio",
        "cwd": SCRIPT_DIR,
    }


def _mcp_tool_text(result: object) -> str:
    """Normalize MCP / LangChain tool return values to a plain string."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        text = result[0].get("text")
        if isinstance(text, str):
            return text
    return str(result)


async def main():
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit(
            "Missing GROQ_API_KEY. Add it to .env in the project folder or set it in the environment."
        )

    model = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    invoke_cfg = {"recursion_limit": 25}

    math_client = MultiServerMCPClient({"math": _stdio_server("math_mcp.py")})
    math_tools = await math_client.get_tools()
    math_agent = create_react_agent(model, math_tools, prompt=MATH_SYSTEM_PROMPT)

    math_response = await math_agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]},
        invoke_cfg,
    )
    print("Math response:", math_response["messages"][-1].content)

    # Groq + single-tool agents often loop or ignore tool output here; call MCP directly.
    weather_client = MultiServerMCPClient({"weather": _stdio_server("weather.py")})
    weather_tools = await weather_client.get_tools()
    get_weather = next(t for t in weather_tools if t.name == "get_weather")
    weather_raw = await get_weather.ainvoke({"location": "California"})
    print("Weather response:", _mcp_tool_text(weather_raw))


asyncio.run(main())
