#!/usr/bin/env python3
"""
Mistral Vibe MCP Tools Server

完整实现的 MCP 服务器，提供以下工具:
- bash: 执行 shell 命令
- grep: 递归搜索文件内容
- read_file: 读取文件内容
- write_file: 写入文件
- search_replace: 搜索替换文件内容
- todo: 任务管理

运行方式:
    python vibe_mcp_tools.py
    # 或使用 uvx
    uvx fastmcp run vibe_mcp_tools.py
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import shutil
import signal
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, NamedTuple
from dotenv import load_dotenv
load_dotenv()
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# ============================================================================
# MCP Server 初始化
# ============================================================================
mcp = FastMCP(name="Mistral Vibe Tools")

# 全局状态存储
_TODO_STORE: list[dict[str, str]] = []
_SEARCH_HISTORY: list[str] = []
_RECENTLY_READ_FILES: list[str] = []
_RECENTLY_WRITTEN_FILES: list[str] = []

# 配置常量
MAX_OUTPUT_BYTES = 64_000
MAX_READ_BYTES = 64_000
MAX_WRITE_BYTES = 64_000
DEFAULT_BASH_TIMEOUT = 30
DEFAULT_GREP_MAX_MATCHES = 100
DEFAULT_GREP_TIMEOUT = 60
MAX_TODOS = 100

# ============================================================================
# 工作目录配置 (安全沙箱)
# ============================================================================
# 通过环境变量配置工作目录，所有文件操作都限制在此目录下
WORKING_DIR = Path(os.getenv("WORKING_DIR", "./workspace")).resolve()
# 资源访问的基础 URL
RESOURCE_BASE_URL = os.getenv("RESOURCE_BASE_URL", "http://localhost:8000/resources")


def _ensure_working_dir() -> Path:
    """确保工作目录存在"""
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    return WORKING_DIR


def _is_path_safe(path: Path) -> bool:
    """检查路径是否在工作目录内（防止路径遍历攻击）"""
    try:
        resolved = path.resolve()
        return resolved == WORKING_DIR or WORKING_DIR in resolved.parents
    except (ValueError, OSError):
        return False


def _resolve_safe_path(
    path_str: str, must_exist: bool = False
) -> tuple[Path | None, str | None]:
    """
    安全地解析路径，确保在工作目录内

    Args:
        path_str: 用户提供的路径
        must_exist: 是否要求路径必须存在

    Returns:
        (resolved_path, error_message) - 成功时 error_message 为 None
    """
    _ensure_working_dir()

    if not path_str.strip():
        return None, "Path cannot be empty"

    user_path = Path(path_str).expanduser()

    # 如果是相对路径，相对于工作目录
    if not user_path.is_absolute():
        resolved = (WORKING_DIR / user_path).resolve()
    else:
        resolved = user_path.resolve()

    # 安全检查
    if not _is_path_safe(resolved):
        return (
            None,
            f"Access denied: path '{path_str}' is outside working directory '{WORKING_DIR}'",
        )

    if must_exist and not resolved.exists():
        return None, f"Path does not exist: {path_str}"

    return resolved, None


def _get_resource_url(file_path: Path) -> str:
    """获取文件的资源访问 URL"""
    try:
        relative_path = file_path.relative_to(WORKING_DIR)
        return f"{RESOURCE_BASE_URL}/{relative_path}"
    except ValueError:
        return ""


# ============================================================================
# 辅助函数
# ============================================================================
def _get_subprocess_encoding() -> str:
    """获取子进程输出编码"""
    if sys.platform == "win32":
        import ctypes

        return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    return "utf-8"


def _is_windows() -> bool:
    """检查是否为 Windows 系统"""
    return sys.platform == "win32"


def _get_base_env() -> dict[str, str]:
    """获取子进程环境变量"""
    base_env = {
        **os.environ,
        "CI": "true",
        "NONINTERACTIVE": "1",
        "NO_TTY": "1",
        "NO_COLOR": "1",
    }

    if _is_windows():
        base_env["GIT_PAGER"] = "more"
        base_env["PAGER"] = "more"
    else:
        base_env["TERM"] = "dumb"
        base_env["DEBIAN_FRONTEND"] = "noninteractive"
        base_env["GIT_PAGER"] = "cat"
        base_env["PAGER"] = "cat"
        base_env["LESS"] = "-FX"
        base_env["LC_ALL"] = "en_US.UTF-8"

    return base_env


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """终止进程树"""
    if proc.returncode is not None:
        return

    try:
        if sys.platform == "win32":
            try:
                subprocess_proc = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(proc.pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await subprocess_proc.wait()
            except (FileNotFoundError, OSError):
                proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        await proc.wait()
    except (ProcessLookupError, PermissionError, OSError):
        pass


# ============================================================================
# Bash Tool - 执行 shell 命令
# ============================================================================
@mcp.tool
async def bash(
    command: str,
    timeout: int | None = None,
    working_directory: str | None = None,
) -> dict[str, str | int]:
    """Run a one-off bash command and capture its output.

    Use this for system operations, git commands, and quick checks.
    DO NOT use for file reading (use read_file), searching (use grep),
    or file editing (use write_file/search_replace).

    Note: Commands are executed within the sandbox working directory for security.

    Args:
        command: The shell command to execute.
        timeout: Override the default command timeout (default: 30s).
        working_directory: Subdirectory within workspace to run the command in.

    Returns:
        dict with stdout, stderr, and returncode.
    """
    effective_timeout = timeout or DEFAULT_BASH_TIMEOUT

    # 确保工作目录在沙箱内
    _ensure_working_dir()
    if working_directory:
        cwd, error = _resolve_safe_path(working_directory)
        if error:
            return {"stdout": "", "stderr": error, "returncode": -1}
    else:
        cwd = WORKING_DIR

    proc = None
    try:
        kwargs: dict[str, Any] = {} if _is_windows() else {"start_new_session": True}

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=str(cwd),
            env=_get_base_env(),
            **kwargs,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout
            )
        except TimeoutError:
            await _kill_process_tree(proc)
            return {
                "stdout": "",
                "stderr": f"Command timed out after {effective_timeout}s: {command!r}",
                "returncode": -1,
            }

        encoding = _get_subprocess_encoding()
        stdout = (
            stdout_bytes.decode(encoding, errors="replace")[:MAX_OUTPUT_BYTES]
            if stdout_bytes
            else ""
        )
        stderr = (
            stderr_bytes.decode(encoding, errors="replace")[:MAX_OUTPUT_BYTES]
            if stderr_bytes
            else ""
        )
        returncode = proc.returncode or 0

        return {"stdout": stdout, "stderr": stderr, "returncode": returncode}

    except Exception as exc:
        return {
            "stdout": "",
            "stderr": f"Error running command {command!r}: {exc}",
            "returncode": -1,
        }
    finally:
        if proc is not None:
            await _kill_process_tree(proc)


# ============================================================================
# Grep Tool - 文件内容搜索
# ============================================================================
# 默认排除的目录和文件
DEFAULT_EXCLUDE_PATTERNS = [
    ".venv/",
    "venv/",
    ".env/",
    "env/",
    "node_modules/",
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".tox/",
    ".nox/",
    ".coverage/",
    "htmlcov/",
    "dist/",
    "build/",
    ".idea/",
    ".vscode/",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
]


class GrepBackend(StrEnum):
    RIPGREP = auto()
    GNU_GREP = auto()


def _detect_grep_backend() -> GrepBackend:
    """检测可用的 grep 后端"""
    if shutil.which("rg"):
        return GrepBackend.RIPGREP
    if shutil.which("grep"):
        return GrepBackend.GNU_GREP
    raise RuntimeError(
        "Neither ripgrep (rg) nor grep is installed. "
        "Please install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
    )


def _build_ripgrep_command(
    pattern: str,
    path: str,
    max_matches: int,
    use_default_ignore: bool,
    exclude_patterns: list[str],
) -> list[str]:
    """构建 ripgrep 命令"""
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--smart-case",
        "--no-binary",
        "--max-count",
        str(max_matches + 1),
    ]

    if not use_default_ignore:
        cmd.append("--no-ignore")

    for pattern_glob in exclude_patterns:
        cmd.extend(["--glob", f"!{pattern_glob}"])

    cmd.extend(["-e", pattern, path])
    return cmd


def _build_gnu_grep_command(
    pattern: str,
    path: str,
    max_matches: int,
    exclude_patterns: list[str],
) -> list[str]:
    """构建 GNU grep 命令"""
    cmd = ["grep", "-r", "-n", "-I", "-E", f"--max-count={max_matches + 1}"]

    if pattern.islower():
        cmd.append("-i")

    for pattern_glob in exclude_patterns:
        if pattern_glob.endswith("/"):
            dir_pattern = pattern_glob.rstrip("/")
            cmd.append(f"--exclude-dir={dir_pattern}")
        else:
            cmd.append(f"--exclude={pattern_glob}")

    cmd.extend(["-e", pattern, path])
    return cmd


@mcp.tool
async def grep(
    pattern: str,
    path: str = ".",
    max_matches: int | None = None,
    use_default_ignore: bool = True,
    working_directory: str | None = None,
) -> dict[str, str | int | bool]:
    """Recursively search files for a regex pattern using ripgrep (rg) or grep.

    Very fast and automatically ignores files like .pyc, .venv directories, etc.
    Use this to find function definitions, variable usage, or locate error messages.

    Note: Search is restricted to the sandbox working directory.

    Args:
        pattern: The regex pattern to search for.
        path: The file or directory path to search in (relative to workspace).
        max_matches: Override the default maximum number of matches (default: 100).
        use_default_ignore: Whether to respect .gitignore and .ignore files (default: True).
        working_directory: Subdirectory within workspace to run the search from.

    Returns:
        dict with matches, match_count, and was_truncated flag.
    """
    if not pattern.strip():
        return {
            "matches": "",
            "match_count": 0,
            "was_truncated": False,
            "error": "Empty search pattern provided.",
        }

    _ensure_working_dir()

    # 解析工作目录
    if working_directory:
        cwd, error = _resolve_safe_path(working_directory)
        if error:
            return {
                "matches": "",
                "match_count": 0,
                "was_truncated": False,
                "error": error,
            }
    else:
        cwd = WORKING_DIR

    effective_max_matches = max_matches or DEFAULT_GREP_MAX_MATCHES

    # 验证搜索路径
    search_path, error = _resolve_safe_path(path if path != "." else str(cwd))
    if error:
        return {"matches": "", "match_count": 0, "was_truncated": False, "error": error}

    if not search_path.exists():
        return {
            "matches": "",
            "match_count": 0,
            "was_truncated": False,
            "error": f"Path does not exist: {path}",
        }

    try:
        backend = _detect_grep_backend()
    except RuntimeError as e:
        return {
            "matches": "",
            "match_count": 0,
            "was_truncated": False,
            "error": str(e),
        }

    # 构建命令
    if backend == GrepBackend.RIPGREP:
        cmd = _build_ripgrep_command(
            pattern,
            str(search_path),
            effective_max_matches,
            use_default_ignore,
            DEFAULT_EXCLUDE_PATTERNS,
        )
    else:
        cmd = _build_gnu_grep_command(
            pattern, str(search_path), effective_max_matches, DEFAULT_EXCLUDE_PATTERNS
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=DEFAULT_GREP_TIMEOUT
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "matches": "",
                "match_count": 0,
                "was_truncated": False,
                "error": f"Search timed out after {DEFAULT_GREP_TIMEOUT}s",
            }

        stdout = stdout_bytes.decode("utf-8", errors="ignore") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="ignore") if stderr_bytes else ""

        if proc.returncode not in {0, 1}:
            error_msg = stderr or f"Process exited with code {proc.returncode}"
            return {
                "matches": "",
                "match_count": 0,
                "was_truncated": False,
                "error": f"grep error: {error_msg}",
            }

        # 解析输出
        output_lines = stdout.splitlines() if stdout else []
        truncated_lines = output_lines[:effective_max_matches]
        truncated_output = "\n".join(truncated_lines)

        was_truncated = (
            len(output_lines) > effective_max_matches
            or len(truncated_output) > MAX_OUTPUT_BYTES
        )

        final_output = truncated_output[:MAX_OUTPUT_BYTES]

        # 记录搜索历史
        _SEARCH_HISTORY.append(pattern)
        if len(_SEARCH_HISTORY) > 50:
            _SEARCH_HISTORY.pop(0)

        return {
            "matches": final_output,
            "match_count": len(truncated_lines),
            "was_truncated": was_truncated,
        }

    except Exception as exc:
        return {
            "matches": "",
            "match_count": 0,
            "was_truncated": False,
            "error": f"Error running grep: {exc}",
        }


# ============================================================================
# ReadFile Tool - 读取文件内容
# ============================================================================
@mcp.tool
async def read_file(
    path: str,
    offset: int = 0,
    limit: int | None = None,
    working_directory: str | None = None,
) -> dict[str, str | int | bool]:
    """Read a UTF-8 file, returning content from a specific line range.

    Designed to handle large files safely with pagination.
    Note: File access is restricted to the sandbox working directory.

    Strategy for large files:
    1. Call read_file with a limit (e.g., 1000 lines) to get the start
    2. If was_truncated is true, the file is large
    3. Read next chunk with offset (e.g., offset=1000, limit=1000)

    Args:
        path: The file path to read (relative to workspace).
        offset: Line number to start reading from (0-indexed, inclusive).
        limit: Maximum number of lines to read.
        working_directory: Subdirectory within workspace for relative paths.

    Returns:
        dict with path, content, lines_read, and was_truncated flag.
    """
    if offset < 0:
        return {
            "path": "",
            "content": "",
            "lines_read": 0,
            "was_truncated": False,
            "error": "Offset cannot be negative",
        }

    if limit is not None and limit <= 0:
        return {
            "path": "",
            "content": "",
            "lines_read": 0,
            "was_truncated": False,
            "error": "Limit must be a positive number",
        }

    _ensure_working_dir()

    # 如果指定了工作子目录，先验证
    if working_directory:
        base_dir, error = _resolve_safe_path(working_directory)
        if error:
            return {
                "path": "",
                "content": "",
                "lines_read": 0,
                "was_truncated": False,
                "error": error,
            }
    else:
        base_dir = WORKING_DIR

    # 解析文件路径
    file_path = Path(path).expanduser()
    if not file_path.is_absolute():
        file_path = base_dir / file_path

    resolved_path, error = _resolve_safe_path(str(file_path), must_exist=True)
    if error:
        return {
            "path": str(file_path),
            "content": "",
            "lines_read": 0,
            "was_truncated": False,
            "error": error,
        }

    if resolved_path.is_dir():
        return {
            "path": str(file_path),
            "content": "",
            "lines_read": 0,
            "was_truncated": False,
            "error": f"Path is a directory, not a file: {file_path}",
        }

    try:
        lines_to_return: list[str] = []
        bytes_read = 0
        was_truncated = False

        with open(resolved_path, encoding="utf-8", errors="ignore") as f:
            for line_index, line in enumerate(f):
                if line_index < offset:
                    continue

                if limit is not None and len(lines_to_return) >= limit:
                    break

                line_bytes = len(line.encode("utf-8"))
                if bytes_read + line_bytes > MAX_READ_BYTES:
                    was_truncated = True
                    break

                lines_to_return.append(line)
                bytes_read += line_bytes

        # 记录读取历史
        _RECENTLY_READ_FILES.append(str(resolved_path))
        if len(_RECENTLY_READ_FILES) > 10:
            _RECENTLY_READ_FILES.pop(0)

        return {
            "path": str(resolved_path),
            "content": "".join(lines_to_return),
            "lines_read": len(lines_to_return),
            "was_truncated": was_truncated,
        }

    except OSError as exc:
        return {
            "path": str(file_path),
            "content": "",
            "lines_read": 0,
            "was_truncated": False,
            "error": f"Error reading {file_path}: {exc}",
        }


# ============================================================================
# WriteFile Tool - 写入文件
# ============================================================================
@mcp.tool
async def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
    create_parent_dirs: bool = True,
    working_directory: str | None = None,
) -> dict[str, str | int | bool]:
    """Create or overwrite a UTF-8 file.

    Fails if file exists unless 'overwrite=True'.
    Note: File writes are restricted to the sandbox working directory.

    Args:
        path: The file path to write to (relative to workspace).
        content: The content to write.
        overwrite: Must be set to True to overwrite an existing file.
        create_parent_dirs: Whether to create parent directories if they don't exist.
        working_directory: Subdirectory within workspace for relative paths.

    Returns:
        dict with path, bytes_written, file_existed, content, and resource_url.
    """
    if not path.strip():
        return {
            "path": "",
            "bytes_written": 0,
            "file_existed": False,
            "content": "",
            "resource_url": "",
            "error": "Path cannot be empty",
        }

    content_bytes = len(content.encode("utf-8"))
    if content_bytes > MAX_WRITE_BYTES:
        return {
            "path": "",
            "bytes_written": 0,
            "file_existed": False,
            "content": "",
            "resource_url": "",
            "error": f"Content exceeds {MAX_WRITE_BYTES} bytes limit",
        }

    _ensure_working_dir()

    # 如果指定了工作子目录，先验证
    if working_directory:
        base_dir, error = _resolve_safe_path(working_directory)
        if error:
            return {
                "path": "",
                "bytes_written": 0,
                "file_existed": False,
                "content": "",
                "resource_url": "",
                "error": error,
            }
    else:
        base_dir = WORKING_DIR

    # 解析文件路径
    file_path = Path(path).expanduser()
    if not file_path.is_absolute():
        file_path = base_dir / file_path

    resolved_path, error = _resolve_safe_path(str(file_path))
    if error:
        return {
            "path": str(file_path),
            "bytes_written": 0,
            "file_existed": False,
            "content": "",
            "resource_url": "",
            "error": error,
        }

    file_existed = resolved_path.exists()

    if file_existed and not overwrite:
        return {
            "path": str(resolved_path),
            "bytes_written": 0,
            "file_existed": True,
            "content": "",
            "resource_url": "",
            "error": f"File '{resolved_path}' exists. Set overwrite=True to replace.",
        }

    if create_parent_dirs:
        # 确保父目录也在工作目录内
        parent_path, parent_error = _resolve_safe_path(str(resolved_path.parent))
        if parent_error:
            return {
                "path": str(resolved_path),
                "bytes_written": 0,
                "file_existed": False,
                "content": "",
                "resource_url": "",
                "error": parent_error,
            }
        parent_path.mkdir(parents=True, exist_ok=True)
    elif not resolved_path.parent.exists():
        return {
            "path": str(resolved_path),
            "bytes_written": 0,
            "file_existed": False,
            "content": "",
            "resource_url": "",
            "error": f"Parent directory does not exist: {resolved_path.parent}",
        }

    try:
        with open(resolved_path, mode="w", encoding="utf-8") as f:
            f.write(content)

        # 记录写入历史
        _RECENTLY_WRITTEN_FILES.append(str(resolved_path))
        if len(_RECENTLY_WRITTEN_FILES) > 10:
            _RECENTLY_WRITTEN_FILES.pop(0)

        # 生成资源 URL
        resource_url = _get_resource_url(resolved_path)

        return {
            "path": str(resolved_path),
            "bytes_written": content_bytes,
            "file_existed": file_existed,
            "content": content,
            "resource_url": resource_url,
        }

    except Exception as e:
        return {
            "path": str(resolved_path),
            "bytes_written": 0,
            "file_existed": file_existed,
            "content": "",
            "resource_url": "",
            "error": f"Error writing {resolved_path}: {e}",
        }


# ============================================================================
# SearchReplace Tool - 搜索替换文件内容
# ============================================================================
_BLOCK_RE = re.compile(
    r"<{5,} SEARCH\r?\n(.*?)\r?\n?={5,}\r?\n(.*?)\r?\n?>{5,} REPLACE", flags=re.DOTALL
)

_BLOCK_WITH_FENCE_RE = re.compile(
    r"```[\s\S]*?\n<{5,} SEARCH\r?\n(.*?)\r?\n?={5,}\r?\n(.*?)\r?\n?>{5,} REPLACE\s*\n```",
    flags=re.DOTALL,
)


class SearchReplaceBlock(NamedTuple):
    search: str
    replace: str


class FuzzyMatch(NamedTuple):
    similarity: float
    start_line: int
    end_line: int
    text: str


def _parse_search_replace_blocks(content: str) -> list[SearchReplaceBlock]:
    """解析 SEARCH/REPLACE 块"""
    matches = _BLOCK_WITH_FENCE_RE.findall(content)

    if not matches:
        matches = _BLOCK_RE.findall(content)

    return [
        SearchReplaceBlock(search=search.rstrip("\r\n"), replace=replace.rstrip("\r\n"))
        for search, replace in matches
    ]


def _find_best_fuzzy_match(
    content: str, search_text: str, threshold: float = 0.9
) -> FuzzyMatch | None:
    """查找最佳模糊匹配"""
    content_lines = content.split("\n")
    search_lines = search_text.split("\n")
    window_size = len(search_lines)

    if window_size == 0:
        return None

    non_empty_search = [line for line in search_lines if line.strip()]
    if not non_empty_search:
        return None

    first_anchor = non_empty_search[0]
    last_anchor = non_empty_search[-1] if len(non_empty_search) > 1 else first_anchor

    candidate_starts = set()
    spread = 5

    for i, line in enumerate(content_lines):
        if first_anchor in line or last_anchor in line:
            start_min = max(0, i - spread)
            start_max = min(len(content_lines) - window_size + 1, i + spread + 1)
            for s in range(start_min, start_max):
                candidate_starts.add(s)

    if not candidate_starts:
        max_positions = min(len(content_lines) - window_size + 1, 100)
        candidate_starts = set(range(0, max_positions))

    best_match = None
    best_similarity = 0.0

    for start in candidate_starts:
        end = start + window_size
        window_text = "\n".join(content_lines[start:end])

        matcher = difflib.SequenceMatcher(None, search_text, window_text)
        similarity = matcher.ratio()

        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = FuzzyMatch(
                similarity=similarity,
                start_line=start + 1,
                end_line=end,
                text=window_text,
            )

    return best_match


def _find_search_context(content: str, search_text: str, max_context: int = 5) -> str:
    """查找搜索上下文"""
    lines = content.split("\n")
    search_lines = search_text.split("\n")

    if not search_lines:
        return "Search text is empty"

    first_search_line = search_lines[0].strip()
    if not first_search_line:
        return "First line of search text is empty or whitespace only"

    matches = []
    for i, line in enumerate(lines):
        if first_search_line in line:
            matches.append(i)

    if not matches:
        return f"First search line '{first_search_line}' not found anywhere in file"

    context_lines = []
    for match_idx in matches[:3]:
        start = max(0, match_idx - max_context)
        end = min(len(lines), match_idx + max_context + 1)

        context_lines.append(f"\nPotential match area around line {match_idx + 1}:")
        for i in range(start, end):
            marker = ">>>" if i == match_idx else "   "
            context_lines.append(f"{marker} {i + 1:3d}: {lines[i]}")

    return "\n".join(context_lines)


@mcp.tool
async def search_replace(
    file_path: str,
    content: str,
    fuzzy_threshold: float = 0.9,
    create_backup: bool = False,
    working_directory: str | None = None,
) -> dict[str, str | int | list[str]]:
    """Replace sections of files using SEARCH/REPLACE blocks.

    Supports fuzzy matching and detailed error reporting.
    Note: File access is restricted to the sandbox working directory.

    Format:
        <<<<<<< SEARCH
        [exact content to find]
        =======
        [new content to replace with]
        >>>>>>> REPLACE

    Args:
        file_path: The file to modify (relative to workspace).
        content: The SEARCH/REPLACE block(s) to apply.
        fuzzy_threshold: Similarity threshold for fuzzy matching (default: 0.9).
        create_backup: Whether to create a .bak backup file.
        working_directory: Subdirectory within workspace for relative paths.

    Returns:
        dict with file, blocks_applied, lines_changed, content, resource_url, and warnings.
    """
    file_path_str = file_path.strip()
    content_str = content.strip()

    if not file_path_str:
        return {
            "file": "",
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": "File path cannot be empty",
        }

    if len(content_str) > 100_000:
        return {
            "file": "",
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": "Content size exceeds 100KB limit",
        }

    if not content_str:
        return {
            "file": "",
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": "Empty content provided",
        }

    _ensure_working_dir()

    # 如果指定了工作子目录，先验证
    if working_directory:
        base_dir, error = _resolve_safe_path(working_directory)
        if error:
            return {
                "file": "",
                "blocks_applied": 0,
                "lines_changed": 0,
                "content": "",
                "resource_url": "",
                "warnings": [],
                "error": error,
            }
    else:
        base_dir = WORKING_DIR

    # 解析文件路径
    target_path = Path(file_path_str).expanduser()
    if not target_path.is_absolute():
        target_path = base_dir / target_path

    resolved_target, error = _resolve_safe_path(str(target_path), must_exist=True)
    if error:
        return {
            "file": str(target_path),
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": error,
        }

    if not resolved_target.is_file():
        return {
            "file": str(resolved_target),
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": f"Path is not a file: {resolved_target}",
        }

    # 解析块
    blocks = _parse_search_replace_blocks(content_str)
    if not blocks:
        return {
            "file": str(resolved_target),
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": (
                "No valid SEARCH/REPLACE blocks found in content.\n"
                "Expected format:\n"
                "<<<<<<< SEARCH\n"
                "[exact content to find]\n"
                "=======\n"
                "[new content to replace with]\n"
                ">>>>>>> REPLACE"
            ),
        }

    # 读取文件
    try:
        with open(resolved_target, encoding="utf-8") as f:
            original_content = f.read()
    except UnicodeDecodeError as e:
        return {
            "file": str(resolved_target),
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": f"Unicode decode error: {e}",
        }
    except PermissionError:
        return {
            "file": str(resolved_target),
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": f"Permission denied reading file: {resolved_target}",
        }
    except Exception as e:
        return {
            "file": str(resolved_target),
            "blocks_applied": 0,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": [],
            "error": f"Error reading file: {e}",
        }

    # 应用块
    applied = 0
    errors: list[str] = []
    warnings: list[str] = []
    current_content = original_content

    for i, (search, replace) in enumerate(blocks, 1):
        if search not in current_content:
            context = _find_search_context(current_content, search)
            fuzzy_match = _find_best_fuzzy_match(
                current_content, search, fuzzy_threshold
            )

            error_msg = (
                f"SEARCH/REPLACE block {i} failed: Search text not found in {resolved_target}\n"
                f"Search text was:\n{search!r}\n"
                f"Context analysis:\n{context}"
            )

            if fuzzy_match:
                similarity_pct = fuzzy_match.similarity * 100
                error_msg += (
                    f"\n\nClosest fuzzy match (similarity {similarity_pct:.1f}%) "
                    f"at lines {fuzzy_match.start_line}–{fuzzy_match.end_line}:\n"
                    f"{fuzzy_match.text}"
                )

            error_msg += (
                "\n\nDebugging tips:\n"
                "1. Check for exact whitespace/indentation match\n"
                "2. Verify line endings match the file exactly (\\r\\n vs \\n)\n"
                "3. Ensure the search text hasn't been modified by previous blocks\n"
                "4. Check for typos or case sensitivity issues"
            )

            errors.append(error_msg)
            continue

        occurrences = current_content.count(search)
        if occurrences > 1:
            warnings.append(
                f"Search text in block {i} appears {occurrences} times. "
                f"Only the first occurrence will be replaced."
            )

        current_content = current_content.replace(search, replace, 1)
        applied += 1

    if errors:
        error_message = "SEARCH/REPLACE blocks failed:\n" + "\n\n".join(errors)
        if warnings:
            error_message += "\n\nWarnings:\n" + "\n".join(warnings)
        return {
            "file": str(resolved_target),
            "blocks_applied": applied,
            "lines_changed": 0,
            "content": "",
            "resource_url": "",
            "warnings": warnings,
            "error": error_message,
        }

    # 计算行变化
    if current_content == original_content:
        lines_changed = 0
    else:
        original_lines = len(original_content.splitlines())
        new_lines = len(current_content.splitlines())
        lines_changed = new_lines - original_lines

        # 创建备份
        if create_backup:
            try:
                shutil.copy2(
                    resolved_target,
                    resolved_target.with_suffix(resolved_target.suffix + ".bak"),
                )
            except Exception:
                pass

        # 写入文件
        try:
            with open(resolved_target, mode="w", encoding="utf-8") as f:
                f.write(current_content)
        except PermissionError:
            return {
                "file": str(resolved_target),
                "blocks_applied": applied,
                "lines_changed": 0,
                "content": "",
                "resource_url": "",
                "warnings": warnings,
                "error": f"Permission denied writing to file: {resolved_target}",
            }
        except Exception as e:
            return {
                "file": str(resolved_target),
                "blocks_applied": applied,
                "lines_changed": 0,
                "content": "",
                "resource_url": "",
                "warnings": warnings,
                "error": f"Error writing file: {e}",
            }

    # 生成资源 URL
    resource_url = _get_resource_url(resolved_target)

    return {
        "file": str(resolved_target),
        "blocks_applied": applied,
        "lines_changed": lines_changed,
        "content": content_str,
        "resource_url": resource_url,
        "warnings": warnings,
    }


# ============================================================================
# Todo Tool - 任务管理
# ============================================================================
class TodoStatus(StrEnum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class TodoPriority(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@mcp.tool
def todo(
    action: str,
    todos: list[dict[str, str]] | None = None,
) -> dict[str, str | list[dict[str, str]] | int]:
    """Manage todos for tracking task progress.

    Args:
        action: Either 'read' or 'write'.
        todos: Complete list of todos when writing. Each todo should have:
            - id: Unique identifier (required)
            - content: Task description (required)
            - status: One of 'pending', 'in_progress', 'completed', 'cancelled' (default: 'pending')
            - priority: One of 'low', 'medium', 'high' (default: 'medium')

    Returns:
        dict with message, todos list, and total_count.
    """
    global _TODO_STORE

    match action:
        case "read":
            return {
                "message": f"Retrieved {len(_TODO_STORE)} todos",
                "todos": _TODO_STORE.copy(),
                "total_count": len(_TODO_STORE),
            }
        case "write":
            if todos is None:
                todos = []

            if len(todos) > MAX_TODOS:
                return {
                    "message": f"Cannot store more than {MAX_TODOS} todos",
                    "todos": _TODO_STORE.copy(),
                    "total_count": len(_TODO_STORE),
                    "error": f"Exceeds maximum of {MAX_TODOS} todos",
                }

            # 验证并规范化 todos
            normalized_todos: list[dict[str, str]] = []
            ids_seen: set[str] = set()

            for t in todos:
                if "id" not in t:
                    return {
                        "message": "Todo missing required 'id' field",
                        "todos": _TODO_STORE.copy(),
                        "total_count": len(_TODO_STORE),
                        "error": "Todo missing required 'id' field",
                    }

                if t["id"] in ids_seen:
                    return {
                        "message": "Todo IDs must be unique",
                        "todos": _TODO_STORE.copy(),
                        "total_count": len(_TODO_STORE),
                        "error": f"Duplicate todo ID: {t['id']}",
                    }
                ids_seen.add(t["id"])

                normalized_todos.append(
                    {
                        "id": t["id"],
                        "content": t.get("content", ""),
                        "status": t.get("status", "pending"),
                        "priority": t.get("priority", "medium"),
                    }
                )

            _TODO_STORE = normalized_todos

            return {
                "message": f"Updated {len(_TODO_STORE)} todos",
                "todos": _TODO_STORE.copy(),
                "total_count": len(_TODO_STORE),
            }
        case _:
            return {
                "message": f"Invalid action '{action}'. Use 'read' or 'write'.",
                "todos": _TODO_STORE.copy(),
                "total_count": len(_TODO_STORE),
                "error": f"Invalid action: {action}",
            }


# ============================================================================
# 额外实用工具
# ============================================================================
@mcp.tool
def list_directory(
    path: str = ".",
    working_directory: str | None = None,
    show_hidden: bool = False,
) -> dict[str, list[str] | str]:
    """List files and directories in a given path.

    Note: Directory listing is restricted to the sandbox working directory.

    Args:
        path: Directory path to list (relative to workspace).
        working_directory: Subdirectory within workspace for relative paths.
        show_hidden: Whether to show hidden files (starting with .).

    Returns:
        dict with files and directories lists.
    """
    _ensure_working_dir()

    # 如果指定了工作子目录，先验证
    if working_directory:
        base_dir, error = _resolve_safe_path(working_directory)
        if error:
            return {"files": [], "directories": [], "error": error}
    else:
        base_dir = WORKING_DIR

    # 解析目标路径
    target_path = Path(path).expanduser()
    if not target_path.is_absolute():
        target_path = base_dir / target_path

    resolved_path, error = _resolve_safe_path(str(target_path), must_exist=True)
    if error:
        return {"files": [], "directories": [], "error": error}

    if not resolved_path.is_dir():
        return {
            "files": [],
            "directories": [],
            "error": f"Path is not a directory: {path}",
        }

    files: list[str] = []
    directories: list[str] = []

    try:
        for item in sorted(resolved_path.iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue

            if item.is_dir():
                directories.append(item.name + "/")
            else:
                files.append(item.name)

        return {"files": files, "directories": directories, "path": str(resolved_path)}

    except PermissionError:
        return {"files": [], "directories": [], "error": f"Permission denied: {path}"}
    except Exception as e:
        return {
            "files": [],
            "directories": [],
            "error": f"Error listing directory: {e}",
        }


@mcp.tool
def get_file_info(
    path: str,
    working_directory: str | None = None,
) -> dict[str, str | int | bool]:
    """Get information about a file or directory.

    Note: File info is restricted to the sandbox working directory.

    Args:
        path: The file or directory path (relative to workspace).
        working_directory: Subdirectory within workspace for relative paths.

    Returns:
        dict with file/directory information and resource_url for files.
    """
    _ensure_working_dir()

    # 如果指定了工作子目录，先验证
    if working_directory:
        base_dir, error = _resolve_safe_path(working_directory)
        if error:
            return {"exists": False, "path": "", "error": error}
    else:
        base_dir = WORKING_DIR

    # 解析目标路径
    target_path = Path(path).expanduser()
    if not target_path.is_absolute():
        target_path = base_dir / target_path

    resolved_path, error = _resolve_safe_path(str(target_path))
    if error:
        return {"exists": False, "path": str(target_path), "error": error}

    if not resolved_path.exists():
        return {
            "exists": False,
            "path": str(resolved_path),
            "error": f"Path does not exist: {path}",
        }

    try:
        stat_info = resolved_path.stat()

        result = {
            "exists": True,
            "path": str(resolved_path),
            "name": resolved_path.name,
            "is_file": resolved_path.is_file(),
            "is_directory": resolved_path.is_dir(),
            "is_symlink": resolved_path.is_symlink(),
            "size_bytes": stat_info.st_size,
            "extension": resolved_path.suffix if resolved_path.is_file() else "",
        }

        # 如果是文件，添加资源 URL
        if resolved_path.is_file():
            result["resource_url"] = _get_resource_url(resolved_path)

        return result

    except PermissionError:
        return {
            "exists": True,
            "path": str(resolved_path),
            "error": f"Permission denied: {path}",
        }
    except Exception as e:
        return {
            "exists": False,
            "path": str(resolved_path),
            "error": f"Error getting file info: {e}",
        }


# ============================================================================
# Web Search Tool - 网络搜索 (使用 DuckDuckGo，无需 API 密钥)
# ============================================================================
DEFAULT_SEARCH_TIMEOUT = 30
DEFAULT_SEARCH_MAX_RESULTS = 5


@mcp.tool
async def web_search(
    query: str,
    max_results: int | None = None,
    timeout: int | None = None,
) -> dict[str, str | int | list[dict[str, str]]]:
    """Search the web using DuckDuckGo (no API key required).

    Use this tool to find up-to-date information from the internet.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).
        timeout: Search timeout in seconds (default: 30).

    Returns:
        dict with query, results list (each with title, url, snippet), and result_count.
    """
    if not query.strip():
        return {
            "query": "",
            "results": [],
            "result_count": 0,
            "error": "Search query cannot be empty",
        }

    effective_max_results = max_results or DEFAULT_SEARCH_MAX_RESULTS
    effective_timeout = timeout or DEFAULT_SEARCH_TIMEOUT

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {
            "query": query,
            "results": [],
            "result_count": 0,
            "error": "duckduckgo-search package not installed. Run: pip install duckduckgo-search",
        }

    try:
        results: list[dict[str, str]] = []

        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=effective_max_results)

            for r in search_results:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                )

        return {
            "query": query,
            "results": results,
            "result_count": len(results),
        }

    except Exception as e:
        return {
            "query": query,
            "results": [],
            "result_count": 0,
            "error": f"Search failed: {e}",
        }


# ============================================================================
# Python Code Executor Tool - Python 代码执行器
# ============================================================================
DEFAULT_PYTHON_TIMEOUT = 30
MAX_PYTHON_OUTPUT = 64_000


@mcp.tool
async def python_exec(
    code: str,
    timeout: int | None = None,
    working_directory: str | None = None,
) -> dict[str, str | int | bool]:
    """Execute Python code and return the output.

        Use this tool to run Python code snippets, perform calculations,
        data processing, or test code logic.

        The code runs in a subprocess within the sandbox working directory.
        For security, network access and file writes outside working directory
        are restricted.

        Args:
            code: The Python code to execute.
            timeout: Maximum execution time in seconds (default: 30).
            working_directory: Subdirectory within workspace to run the code in.

        Returns:
            dict with stdout, stderr, returncode, and success flag.

        Examples:
            # Simple calculation
            python_exec("print(2 + 2)")

            # Data processing
            python_exec('''
    import json
    data = {"name": "test", "value": 42}
    print(json.dumps(data, indent=2))
            ''')

            # List comprehension
            python_exec("print([x**2 for x in range(10)])")
    """
    if not code.strip():
        return {
            "stdout": "",
            "stderr": "No code provided",
            "returncode": -1,
            "success": False,
        }

    effective_timeout = timeout or DEFAULT_PYTHON_TIMEOUT

    _ensure_working_dir()

    # 解析工作目录
    if working_directory:
        cwd, error = _resolve_safe_path(working_directory)
        if error:
            return {"stdout": "", "stderr": error, "returncode": -1, "success": False}
    else:
        cwd = WORKING_DIR

    proc = None
    try:
        # 使用 -c 参数直接执行代码
        kwargs: dict[str, Any] = {} if _is_windows() else {"start_new_session": True}

        proc = await asyncio.create_subprocess_exec(
            sys.executable,  # 使用当前 Python 解释器
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=str(cwd),
            env=_get_base_env(),
            **kwargs,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout
            )
        except TimeoutError:
            await _kill_process_tree(proc)
            return {
                "stdout": "",
                "stderr": f"Code execution timed out after {effective_timeout}s",
                "returncode": -1,
                "success": False,
            }

        encoding = _get_subprocess_encoding()
        stdout = (
            stdout_bytes.decode(encoding, errors="replace")[:MAX_PYTHON_OUTPUT]
            if stdout_bytes
            else ""
        )
        stderr = (
            stderr_bytes.decode(encoding, errors="replace")[:MAX_PYTHON_OUTPUT]
            if stderr_bytes
            else ""
        )
        returncode = proc.returncode or 0

        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "success": returncode == 0,
        }

    except Exception as exc:
        return {
            "stdout": "",
            "stderr": f"Error executing Python code: {exc}",
            "returncode": -1,
            "success": False,
        }
    finally:
        if proc is not None:
            await _kill_process_tree(proc)


# ============================================================================
# Text to Image Tool - 文字生成图片 (Gemini)
# ============================================================================
T2I_BASEURL = os.getenv("T2I_BASEURL")
T2I_API_KEY = os.getenv("T2I_API_KEY")


@mcp.tool
async def text_to_image(
    prompt: str,
) -> dict[str, str]:
    """Generate an image from text description using Gemini.

    Use this tool to create images based on text prompts.
    The generated image URL can be directly shared with users.

    Args:
        prompt: A detailed description of the image you want to generate.
                Be specific about style, colors, composition, etc.

    Returns:
        dict with image_url on success, or error message on failure.

    Examples:
        text_to_image("一只可爱的橘猫坐在窗台上看夕阳")
        text_to_image("A futuristic city skyline at night with neon lights")
    """
    if not prompt.strip():
        return {"image_url": "", "error": "Prompt cannot be empty"}

    if not T2I_BASEURL or not T2I_API_KEY:
        return {"image_url": "", "error": "T2I_BASEURL or T2I_API_KEY not configured"}

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=T2I_BASEURL, api_key=T2I_API_KEY)
        payload = {
            "model": "gemini-3-pro-imagen",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        response = await client.chat.completions.create(**payload)
        content = response.choices[0].message.content

        # 解析图片 URL: ![image](https://xxx.jpg)
        match = re.search(r'!\[image\]\((https?://[^\)]+)\)', content)
        if not match:
            return {"image_url": "", "error": f"No image URL found in response: {content[:200]}"}

        return {"image_url": match.group(1)}

    except Exception as e:
        return {"image_url": "", "error": f"Image generation failed: {e}"}


# ============================================================================
# 运行服务器
# ============================================================================
if __name__ == "__main__":
    # Default: STDIO transport
    mcp.run()
