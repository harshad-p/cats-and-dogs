import re
import time
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_nebius import ChatNebius
from dotenv import load_dotenv

load_dotenv()


class TimerCallbackHandler(BaseCallbackHandler):
    """Record wall-clock durations for LLM / tool runs."""

    def __init__(self) -> None:
        super().__init__()
        self._starts: dict[Any, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []

    def _extract_name(self, serialized: dict[str, Any] | None) -> str:
        if not serialized:
            return "unknown"
        if isinstance(serialized.get("name"), str):
            return serialized["name"]
        id_parts = serialized.get("id")
        if isinstance(id_parts, list) and id_parts:
            return id_parts[-1]
        return str(serialized)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._starts[run_id] = {
            "kind": "llm",
            "name": self._extract_name(serialized),
            "start": time.monotonic(),
        }

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        rec = self._starts.pop(run_id, None)
        if rec:
            rec["duration"] = time.monotonic() - rec["start"]
            self.records.append(rec)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._starts[run_id] = {
            "kind": "tool",
            "name": self._extract_name(serialized),
            "start": time.monotonic(),
        }

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        rec = self._starts.pop(run_id, None)
        if rec:
            rec["duration"] = time.monotonic() - rec["start"]
            self.records.append(rec)


def extract_clean_markdown(text: str) -> str:
    """Strip model/tool artifacts and return only the markdown-report section."""
    # If the model wraps the report in a fake <content> tag, keep the payload as-is.
    wrapped = False
    marker = "<content>"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx + len(marker):]
        wrapped = True

    # Without a wrapper, the report usually starts at the first markdown heading.
    if not wrapped:
        match = re.search(r"^# .+", text, flags=re.MULTILINE)
        if match:
            text = text[match.start():]

    # Drop any stray artifact tags or lines.
    text = re.sub(r"<[/]?content>", "", text)
    lines = [
        line.lstrip()
        for line in text.splitlines()
        if not line.strip().startswith(("<tool_call", "[<]minimax", "]<]minimax"))
    ]
    return "\n".join(lines).strip()


def find_best_markdown(messages: list[Any]) -> str:
    """Choose the longest message content that contains a markdown heading."""
    best = ""
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and "# " in content and len(content) > len(best):
            best = content
    if not best:
        raise RuntimeError("No markdown content found in agent result")
    return best


def _get_message_tokens(msg: Any) -> tuple[int, int]:
    """Extract input/output tokens from a message's usage metadata."""
    usage = getattr(msg, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    if usage is not None:
        return int(getattr(usage, "input_tokens", 0) or 0), int(
            getattr(usage, "output_tokens", 0) or 0
        )

    response_metadata = getattr(msg, "response_metadata", {}) or {}
    token_usage = (
        response_metadata.get("token_usage")
        if isinstance(response_metadata, dict)
        else None
    )
    if isinstance(token_usage, dict):
        return int(token_usage.get("prompt_tokens") or 0), int(
            token_usage.get("completion_tokens") or 0
        )

    return 0, 0


def print_run_summary(
    messages: list[Any],
    timer: TimerCallbackHandler | None = None,
    total_time: float | None = None,
) -> dict[str, int]:
    """Print a per-call and aggregate token/time summary for the agent run."""
    print("\n--- Run Summary ---")
    print(f"{'Call #':<8} {'Tool calls':<12} {'Input tokens':<14} {'Output tokens':<15}")
    print("-" * 51)

    agent_calls = 0
    total_tool_calls = 0
    total_input = 0
    total_output = 0

    for msg in messages:
        if getattr(msg, "type", None) != "ai":
            continue

        agent_calls += 1
        n_tools = len(getattr(msg, "tool_calls", []) or [])
        inp, out = _get_message_tokens(msg)

        total_tool_calls += n_tools
        total_input += inp
        total_output += out

        print(f"{agent_calls:<8} {n_tools:<12} {inp:<14,} {out:<15,}")

    total_tokens = total_input + total_output
    print("-" * 51)
    print(f"Total agent calls:   {agent_calls}")
    print(f"Total tool calls:    {total_tool_calls}")
    print(f"Total input tokens:  {total_input:,}")
    print(f"Total output tokens: {total_output:,}")
    print(f"Total tokens:        {total_tokens:,}")

    if timer is not None and timer.records:
        print("\n--- Timing Summary ---")
        print(f"{'Step':<6} {'Type':<10} {'Name':<20} {'Duration (s)':<14}")
        print("-" * 54)
        for i, rec in enumerate(timer.records, start=1):
            print(
                f"{i:<6} {rec['kind']:<10} {rec['name']:<20} "
                f"{rec['duration']:<14.3f}"
            )
        print("-" * 54)

    if total_time is not None:
        print(f"Total wall-clock time: {total_time:.3f}s")

    return {
        "agent_calls": agent_calls,
        "tool_calls": total_tool_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
    }


def main() -> None:
    model = ChatNebius(model="MiniMaxAI/MiniMax-M3")
    agent = create_deep_agent(model=model)

    timer = TimerCallbackHandler()

    t0 = time.monotonic()
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Research Solar power adoption in US and write a report"}
            ]
        },
        config={"callbacks": [timer]},
    )
    total_time = time.monotonic() - t0

    messages = result.get("messages", [])
    raw_markdown = find_best_markdown(messages)
    markdown = extract_clean_markdown(raw_markdown)

    print_run_summary(messages, timer=timer, total_time=total_time)
    
    output_path = Path("output.md")
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote report ({len(markdown)} chars) to {output_path}")




if __name__ == "__main__":
    main()
