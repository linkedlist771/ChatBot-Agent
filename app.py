# app.py
"""
OpenAI Compatible API Server with MCP Tools

启动方式:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

使用方式 (兼容 OpenAI API):
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": true
      }'

环境变量配置:
    WORKING_DIR: 工作目录路径 (默认: ./workspace)
    RESOURCE_BASE_URL: 资源访问基础 URL (默认: http://localhost:8000/resources)
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from openai import AsyncOpenAI


# ============================================================================
# 工作目录配置
# ============================================================================
WORKING_DIR = Path(os.getenv("WORKING_DIR", "./workspace")).resolve()
RESOURCE_BASE_URL = os.getenv("RESOURCE_BASE_URL", "http://localhost:8000/resources")


def ensure_working_dir() -> Path:
    """确保工作目录存在"""
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    return WORKING_DIR


def get_system_prompt() -> str:
    """生成系统提示词，告知 Agent 工作环境"""
    return f"""You are a helpful AI assistant with access to various tools for file operations, code execution, and web search.

## Working Environment

You are working in a sandboxed environment:
- All file operations are restricted to a sandbox directory.
- Resource URL Base: {RESOURCE_BASE_URL}

## Important Rules

1. File Access: You can only read, write, and modify files within the sandbox working directory.

2. File URLs: When you create or modify files using write_file or search_replace tools, the result will include a resource_url field. This URL allows the user to download or view the file directly.

3. Final Response: When you complete a task that generates files, you MUST include the resource_url in your final response so the user can access/download the files.

4. Multiple Files: If you create multiple files, list all their URLs at the end of your response.

5. Relative Paths: When using file tools, use relative paths (e.g., output/result.html) rather than absolute paths.

## Response Format

IMPORTANT: Your responses will be displayed in QQ messenger, which does NOT support Markdown rendering.

- Use PLAIN TEXT only, no Markdown syntax
- Do NOT use: **bold**, *italic*, `code`, ```code blocks```, [links](url), # headers, - bullet points
- For emphasis, use CAPS or add spaces like: 重 要 提 示
- For lists, use simple numbering: 1. 2. 3. or Chinese: 一、二、三、
- For code, just paste it directly without backticks
- For URLs, paste the full URL directly, the user can copy it
- Use blank lines to separate paragraphs for readability

Example good response:
我已经为你创建好文件了。

下载地址：
{RESOURCE_BASE_URL}/example.html

如果需要修改，请告诉我。

Remember: Keep responses clean and readable in plain text format."""


# ============================================================================
# Pydantic Models (OpenAI Compatible)
# ============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Usage] = None


# ============================================================================
# MCP Tool Helpers (从 agent.py 复用)
# ============================================================================
def mcp_tool_to_openai_tool(tool) -> Dict[str, Any]:
    """MCP tool -> OpenAI tool format"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def mcp_result_to_text(result: Any) -> str:
    """Convert MCP tool result to text"""
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


# ============================================================================
# MCP Agent (简化版，用于 API)
# ============================================================================
class MCPAgentAPI:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_tools: List[Dict[str, Any]] = []

        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")

        self.llm = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def connect_stdio_server(self, server_script_path: str):
        """连接 MCP Server"""
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

        # 加载工具
        tools_resp = await self.session.list_tools()
        self.openai_tools = [mcp_tool_to_openai_tool(t) for t in tools_resp.tools]
        print(f"[MCP] Loaded {len(self.openai_tools)} tools")

    async def run_stream(
        self, messages: List[Dict[str, Any]], model: str = None
    ) -> AsyncGenerator[str, None]:
        """流式运行 Agent"""
        assert self.session is not None

        use_model = model or self.model
        
        # 注入系统提示词
        system_prompt = get_system_prompt()
        working_messages = [{"role": "system", "content": system_prompt}]
        working_messages.extend([
            {"role": m["role"], "content": m["content"]} for m in messages
        ])

        # Tool-calling loop
        for _ in range(10):
            stream = await self.llm.chat.completions.create(
                model=use_model,
                messages=working_messages,
                tools=self.openai_tools if self.openai_tools else None,
                tool_choice="auto" if self.openai_tools else None,
                stream=True,
            )

            content_parts = []
            tool_calls_data = {}

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                if delta.content:
                    content_parts.append(delta.content)
                    yield delta.content

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

            # No tool call -> done
            if not tool_calls_data:
                return

            # Execute tool calls
            tool_results = []
            for idx in sorted(tool_calls_data.keys()):
                tc = tool_calls_data[idx]
                fn_name = tc["name"]
                fn_args = json.loads(tc["arguments"] or "{}")

                print(f"[Tool Call] {fn_name}({fn_args})")
                result = await self.session.call_tool(fn_name, fn_args)
                result_text = mcp_result_to_text(result)
                print(f"[Tool Result] {result_text[:200]}...")
                tool_results.append(f"Tool '{fn_name}' result: {result_text}")

            # Append to messages
            tool_call_desc = ", ".join(
                [f"{tc['name']}({tc['arguments']})" for tc in tool_calls_data.values()]
            )
            working_messages.append(
                {
                    "role": "assistant",
                    "content": content
                    if content
                    else f"I'll call the tool: {tool_call_desc}",
                }
            )
            working_messages.append(
                {
                    "role": "user",
                    "content": "\n".join(tool_results),
                }
            )

        yield "\n[Reached max tool iterations]"

    async def run(self, messages: List[Dict[str, Any]], model: str = None) -> str:
        """非流式运行 Agent"""
        parts = []
        async for chunk in self.run_stream(messages, model):
            parts.append(chunk)
        return "".join(parts)

    async def aclose(self):
        await self.exit_stack.aclose()


# ============================================================================
# Global Agent Instance
# ============================================================================
agent: Optional[MCPAgentAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global agent

    # 确保工作目录存在
    ensure_working_dir()
    print(f"[Startup] Working directory: {WORKING_DIR}")
    print(f"[Startup] Resource base URL: {RESOURCE_BASE_URL}")

    agent = MCPAgentAPI()

    # 默认连接 tools.py
    mcp_server = os.getenv("MCP_SERVER_PATH", "tools.py")
    try:
        await agent.connect_stdio_server(mcp_server)
        print(f"[Startup] Connected to MCP server: {mcp_server}")
    except Exception as e:
        print(f"[Startup] Failed to connect MCP server: {e}")

    yield

    # Cleanup
    if agent:
        await agent.aclose()
        print("[Shutdown] Agent closed")


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="OpenAI Compatible MCP Agent API",
    description="OpenAI 兼容的 API，集成 MCP 工具，支持沙箱文件操作",
    version="1.0.0",
    lifespan=lifespan,
)

# 挂载静态文件目录，用于访问工作目录中的文件
# 确保工作目录存在
ensure_working_dir()
app.mount("/resources", StaticFiles(directory=str(WORKING_DIR)), name="resources")


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": agent.model if agent else "gpt-4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mcp-agent",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的 Chat Completions API"""
    if not agent or not agent.session:
        raise HTTPException(status_code=503, detail="MCP Agent not initialized")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        # 流式响应
        async def generate_stream():
            async for chunk in agent.run_stream(messages, request.model):
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            # Final chunk
            final_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # 非流式响应
        content = await agent.run(messages, request.model)

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=0,  # 简化处理
                completion_tokens=0,
                total_tokens=0,
            ),
        )


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "mcp_connected": agent is not None and agent.session is not None,
        "tools_count": len(agent.openai_tools) if agent else 0,
        "working_dir": str(WORKING_DIR),
        "resource_base_url": RESOURCE_BASE_URL,
    }


@app.get("/workspace/info")
async def workspace_info():
    """获取工作空间信息"""
    files = []
    directories = []

    try:
        for item in WORKING_DIR.iterdir():
            if item.is_dir():
                directories.append(item.name)
            else:
                files.append(
                    {
                        "name": item.name,
                        "size": item.stat().st_size,
                        "url": f"{RESOURCE_BASE_URL}/{item.name}",
                    }
                )
    except Exception as e:
        return {"error": str(e)}

    return {
        "working_dir": str(WORKING_DIR),
        "resource_base_url": RESOURCE_BASE_URL,
        "files": files,
        "directories": directories,
    }


# ============================================================================
# 直接运行
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
