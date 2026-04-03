from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers. Pass plain numbers only, e.g. a=3, b=5 (not nested objects)."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers. Pass plain numbers only, e.g. a=8, b=12 (not nested objects)."""
    return a * b


@mcp.tool()
def add_then_multiply(a: int, b: int, c: int) -> int:
    """Compute (a + b) * c in one step. Example: for (3 + 5) × 12 use a=3, b=5, c=12."""
    return (a + b) * c

#The transport="stdio" argument tells the server to:

#Use standard input/output (stdin and stdout) to receive and respond to tool function calls.

if __name__=="__main__":
    mcp.run(transport="stdio")