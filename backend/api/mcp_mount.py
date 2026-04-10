"""
Mount the MCP server as an SSE endpoint on the FastAPI app.

Provides remote access to email-and-document-search MCP tools via HTTP/SSE.
Authentication via Bearer token with per-key account restrictions.

Security layers:
1. Tailscale (network-level access control)
2. HTTPS (TLS termination via Tailscale serve)
3. Bearer token (per-user API keys with account restrictions)
"""

import logging
import os
from contextvars import ContextVar
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp.server.sse import SseServerTransport

logger = logging.getLogger(__name__)

# API key → allowed accounts mapping
MCP_API_KEYS = {}

# Context variables for per-request restrictions (thread/async-safe)
_current_allowed_account: ContextVar[Optional[str]] = ContextVar("_current_allowed_account", default=None)
_current_allowed_tools: ContextVar[Optional[str]] = ContextVar("_current_allowed_tools", default=None)


def get_mcp_allowed_account_override() -> Optional[str]:
    """Get the per-request account restriction (if set by SSE endpoint)."""
    return _current_allowed_account.get(None)


def get_mcp_allowed_tools_override() -> Optional[str]:
    """Get the per-request tool restriction (if set by SSE endpoint).

    Returns None (all tools), or a comma-separated string of tool groups:
    'email,applications,listings' etc.
    """
    return _current_allowed_tools.get(None)


def _load_api_keys():
    """Load API keys from environment.

    Keys are configured as:
      MCP_API_KEY=<admin_key>                    (existing, full access)
      MCP_REMOTE_KEYS=<key1>:name1:accounts1:tools1;<key2>:name2:accounts2:tools2

    Tool groups: email, applications, listings, utility (or 'all')
    """
    global MCP_API_KEYS

    # Admin key (existing, backwards compatible) — all accounts, all tools
    admin_key = os.getenv("MCP_API_KEY")
    if admin_key:
        all_accounts = os.getenv("MCP_ALLOWED_ACCOUNT", "personal,work")
        MCP_API_KEYS[admin_key] = {
            "name": "admin",
            "accounts": all_accounts,
            "tools": "all",
        }

    # Remote keys with restricted access
    # Format: key:name:accounts:tools  (tools is optional, defaults to 'email,applications,utility')
    remote_keys = os.getenv("MCP_REMOTE_KEYS", "")
    for entry in remote_keys.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 3)
        if len(parts) >= 3:
            key = parts[0].strip()
            name = parts[1].strip()
            accounts = parts[2].strip()
            tools = parts[3].strip() if len(parts) > 3 else "email,applications,utility"
            MCP_API_KEYS[key] = {
                "name": name,
                "accounts": accounts,
                "tools": tools,
            }
            logger.info(f"Loaded remote MCP key: {name} (accounts: {accounts}, tools: {tools})")
        else:
            logger.warning(f"Invalid MCP_REMOTE_KEYS entry: {entry}")


def _authenticate_request(request: Request) -> Optional[dict]:
    """Validate Bearer token and return key info, or None if invalid."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    return MCP_API_KEYS.get(token)


def create_mcp_sse_app() -> Starlette:
    """Create a Starlette app that serves MCP over SSE with Bearer auth."""
    _load_api_keys()

    if not MCP_API_KEYS:
        logger.warning("No MCP API keys configured — remote MCP endpoint disabled")
        return None

    # Create MCP server
    from mcp_server.server import create_server
    server = create_server()

    # SSE transport — endpoint is relative to mount point
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        """SSE endpoint — client connects here for the event stream."""
        key_info = _authenticate_request(request)
        if not key_info:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        logger.info(f"MCP SSE connection from: {key_info['name']} (accounts: {key_info['accounts']})")

        # Set per-request restrictions via contextvars (async-safe)
        acct_token = _current_allowed_account.set(key_info["accounts"])
        tools_token = _current_allowed_tools.set(key_info.get("tools", "all"))
        try:
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
        finally:
            _current_allowed_account.reset(acct_token)
            _current_allowed_tools.reset(tools_token)

    async def handle_messages(request: Request):
        """POST endpoint — client sends messages here. Authenticated."""
        key_info = _authenticate_request(request)
        if not key_info:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        # Set per-request restrictions
        acct_token = _current_allowed_account.set(key_info["accounts"])
        tools_token = _current_allowed_tools.set(key_info.get("tools", "all"))
        try:
            await sse_transport.handle_post_message(
                request.scope, request.receive, request._send
            )
        finally:
            _current_allowed_account.reset(acct_token)
            _current_allowed_tools.reset(tools_token)

    # Build Starlette app — BOTH routes are authenticated
    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages/{path:path}", endpoint=handle_messages, methods=["POST"]),
        ],
    )

    logger.info(f"MCP SSE endpoint ready ({len(MCP_API_KEYS)} API keys loaded)")
    return app
