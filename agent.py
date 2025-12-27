# agent.py
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()
from contextlib import AsyncExitStack
from typing import Any, AsyncGenerator, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AsyncOpenAI


def mcp_tool_to_openai_tool(tool) -> Dict[str, Any]:
    """
    MCP tool: { name, description, inputSchema }
    OpenAI tool: { type: "function", function: { name, description, parameters } }
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def mcp_result_to_text(result: Any) -> str:
    # MCP call_tool result usually has .content (blocks). Be defensive.
    try:
        blocks = getattr(result, "content", None)
        if isinstance(blocks, list) and blocks:
            parts = []
            for b in blocks:
                if isinstance(b, dict):
                    parts.append(b.get("text") or json.dumps(b, ensure_ascii=False))
                else:
                    parts.append(getattr(b, "text", None) or str(b))
            return "\n".join(parts)
    except Exception:
        pass

    try:
        return json.dumps(result.model_dump(), ensure_ascii=False)
    except Exception:
        return str(result)


class MCPAgent:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL")

        self.llm = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def connect_stdio_server(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        command = "python" if is_python else "node"

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

    async def run(self, user_query: str) -> AsyncGenerator[str, None]:
        assert self.session is not None

        # Load MCP tools
        tools_resp = await self.session.list_tools()
        openai_tools = [mcp_tool_to_openai_tool(t) for t in tools_resp.tools]

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a helpful agent. Use tools when needed.",
            },
            {"role": "user", "content": user_query},
        ]

        # Tool-calling loop
        for _ in range(20):
            # Use async streaming
            stream = await self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                stream=True,
            )

            # Collect streamed response
            content_parts = []
            tool_calls_data = {}  # {index: {id, name, arguments}}

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Collect content
                if delta.content:
                    content_parts.append(delta.content)
                    # print(delta.content, end="", flush=True)
                    yield delta.content
                # Collect tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_data[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += (
                                    tc.function.arguments
                                )

            content = "".join(content_parts)

            # No tool call -> final answer
            if not tool_calls_data:
                if content_parts:
                    print()  # newline after streaming
                return  # Exit the generator when no more tool calls

            # We have tool calls
            print(
                f"\n[Tool calls detected: {[tc['name'] for tc in tool_calls_data.values()]}]"
            )

            # Execute each tool call via MCP, then append tool results
            tool_results = []
            for idx in sorted(tool_calls_data.keys()):
                tc = tool_calls_data[idx]
                fn_name = tc["name"]
                fn_args = json.loads(tc["arguments"] or "{}")

                print(f"[Calling tool: {fn_name} with args: {fn_args}]")
                result = await self.session.call_tool(fn_name, fn_args)
                result_text = mcp_result_to_text(result)
                print(f"[Tool result: {result_text}]")
                tool_results.append(f"Tool '{fn_name}' result: {result_text}")

            # Append assistant message with tool info, then tool results as user message
            # (Claude proxy requires non-empty content)
            tool_call_desc = ", ".join(
                [f"{tc['name']}({tc['arguments']})" for tc in tool_calls_data.values()]
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": content
                    if content
                    else f"I'll call the tool: {tool_call_desc}",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(tool_results),
                }
            )

        yield "Reached max tool iterations."

    async def aclose(self):
        await self.exit_stack.aclose()


async def main():
    import sys

    prompt = "运行下python里面的tricky代码看看？"
    if len(sys.argv) < 2:
        print("Usage: python agent.py path/to/mcp_server.py")
        raise SystemExit(2)

    server_path = sys.argv[1]
    agent = MCPAgent()
    try:
        await agent.connect_stdio_server(server_path)
        async for answer in agent.run(prompt):
            print(answer, end="", flush=True)
        print()
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
