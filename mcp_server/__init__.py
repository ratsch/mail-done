"""
Email Search MCP Server

Exposes semantic and sender-based email search as MCP tools
for integration with Cursor AI and other MCP-compatible clients.

Architecture:
    Cursor → MCP Server → Backend API → Database
    
The MCP server is a thin client that calls the FastAPI backend.
It does NOT access the database directly.
"""
# Import API client (can be used standalone for testing)
from mcp_server.api_client import EmailAPIClient

# Lazy import server functions (requires MCP SDK)
def create_server():
    """Create the MCP server (requires mcp package)."""
    from mcp_server.server import create_server as _create_server
    return _create_server()

def run_server():
    """Run the MCP server (requires mcp package)."""
    from mcp_server.server import run_server as _run_server
    return _run_server()

__all__ = ['EmailAPIClient', 'create_server', 'run_server']
