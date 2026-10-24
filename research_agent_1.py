import re
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_nebius import ChatNebius
from dotenv import load_dotenv

load_dotenv()


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


def main() -> None:
    model = ChatNebius(model="MiniMaxAI/MiniMax-M3")
    agent = create_deep_agent(model=model)

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Research EVs available in US in 2026 and write a report"}
        ]
    })

    raw_markdown = find_best_markdown(result.get("messages", []))
    markdown = extract_clean_markdown(raw_markdown)

    output_path = Path("output.md")
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote report ({len(markdown)} chars) to {output_path.resolve()}")


if __name__ == "__main__":
    main()
