# Live Fact Checker

A voice-first fact checker. Speak into the microphone, and the app transcribes what you say, pulls out factual claims, researches them, and shows a verdict with evidence and sources.

Each claim card shows the original sentence next to the extracted claim. The top bar counts verdicts as checks finish. **Download report** saves a PDF of the current session.

## How it works

| Step | What runs |
|---|---|
| Speech-to-text | Groq Whisper (`whisper-large-v3`) when `GROQ_API_KEY` is set. Otherwise the browser Web Speech API (Chrome or Edge). |
| Claim extraction and verdicts | NVIDIA Nemotron (`nvidia/Nemotron-3_5-Lightning`) via Nebius Token Factory |
| Web research | MiniMax-M3 (`MiniMaxAI/MiniMax-M3`) via Nebius Token Factory and the deepagents framework |
| Sources | Tavily search. Article titles and URLs come from the search results. |

Verdicts are **Supported**, **Contradicted**, **Misleading**, or **Insufficient evidence**. Checks that are still running show as pending. Failed checks stay separate and are not treated as finished verdicts.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A browser with microphone access. Chrome or Edge is required only if you are not using Groq.
- A [Nebius Token Factory](https://tokenfactory.nebius.com/) API key
- A [Tavily](https://tavily.com/) API key
- Optional: a [Groq](https://console.groq.com/) API key for server-side transcription

## Setup

From this directory:

```bash
cd agents/langchain/deep-agent-example-1
cp env.example .env
```

Edit `.env`:

```env
NEBIUS_API_KEY=your-nebius-api-key
TAVILY_API_KEY=your-tavily-api-key

# Optional. Enables Groq Whisper instead of the browser speech API.
GROQ_API_KEY=your-groq-api-key
```

Install dependencies:

```bash
uv sync
```

API keys stay on the server. Do not commit `.env`.

## Run

```bash
uv run python app.py
```

Open http://localhost:5050, click **Start**, and speak a factual claim. Use **Reset** to clear the session.

To use another port:

```bash
PORT=8080 uv run python app.py
```

## Tests

```bash
uv run python -m unittest test_verdict_parsing.py
```
