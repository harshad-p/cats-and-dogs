"""Deep agent with real-time web search via Tavily, on a Nebius Token Factory model.

Same deep-agent loop as `research_agent_1.py` (planning, virtual file system,
sub-agents), but with a Tavily search tool so the agent can research live sources
instead of relying on the model's own knowledge.

Requires `NEBIUS_API_KEY` and `TAVILY_API_KEY` in the environment or in a local
`.env` file — see `env.example`.
"""

import argparse
import json
import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch

load_dotenv()

RESEARCH_QUESTION = (
    "Verify this claim: 'Munich is the capital of Germany.' "
    "Find reliable sources, determine whether the evidence supports "
    "or contradicts the claim, and cite the sources. "
    "If the evidence is insufficient, say so."
)


def require_keys() -> None:
    """Fail early with an actionable message rather than deep inside a client."""
    missing = [
        name
        for name in ("NEBIUS_API_KEY", "TAVILY_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            f"Set {' and '.join(missing)} in the environment or this folder's "
            ".env file (copy env.example to .env)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep research agent — fact-checks a claim using live web search."
    )
    parser.add_argument(
        "--claim",
        type=str,
        default=None,
        help="The claim to verify. Defaults to the built-in example if omitted.",
    )
    return parser.parse_args()


def build_question(claim: str | None) -> str:
    if claim is None:
        return RESEARCH_QUESTION
    return (
        f"Verify this claim: '{claim}' "
        "Find reliable sources, determine whether the evidence supports "
        "or contradicts the claim, and cite the sources. "
        "If the evidence is insufficient, say so."
    )


# Injected before the deep-agent base prompt so the model always runs at least
# one web search instead of answering from parametric knowledge.
SEARCH_FIRST_PROMPT = (
    "You are a fact-checking assistant. "
    "You MUST call the tavily_search tool at least once before writing any conclusion. "
    "Never rely solely on your training knowledge to verify a claim."
)


def main() -> None:
    args = parse_args()
    require_keys()

    question = build_question(args.claim)

    tavily_search = TavilySearch(max_results=5, search_depth="advanced")
    model = ChatNebius(model="MiniMaxAI/MiniMax-M3")
    agent = create_deep_agent(
        model=model,
        tools=[tavily_search],
        system_prompt=SEARCH_FIRST_PROMPT,
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # Full trace first (planning steps, sub-agent calls), then the final report.
    print(json.dumps(result, indent=2, default=str))
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
