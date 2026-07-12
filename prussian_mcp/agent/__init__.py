"""In-process Haystack tools (default) and generator/agent glue for the
``prussian-agent`` CLI.

The agent runs a single ``Agent.run()`` with a system prompt that requires
the model to call ``validate_prussian`` on its draft before printing the
final ``PRUSSIAN:`` line — there is no external orchestration loop.
"""

from .generators import DeepSeekChatGenerator  # noqa: F401  (re-export)
from .runner import (  # noqa: F401  (re-export)
    RunResult,
    build_agent,
    build_generator,
    extract_candidate,
    make_stream_callback,
    parse_last_validation,
    run_agent,
    trace_messages,
)
from .tools import build_local_toolset  # noqa: F401  (re-export)

__all__ = [
    "DeepSeekChatGenerator",
    "RunResult",
    "build_agent",
    "build_generator",
    "build_local_toolset",
    "extract_candidate",
    "make_stream_callback",
    "parse_last_validation",
    "run_agent",
    "trace_messages",
]