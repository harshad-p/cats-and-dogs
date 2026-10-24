# Live Fact Checker

A voice-first fact checker. Speak into the microphone, and the app transcribes what you say, pulls out factual claims, researches them, and shows a verdict with evidence and sources.

Each claim card shows the original sentence next to the extracted claim. The top bar counts verdicts as checks finish. **Download report** saves a PDF of the current session.

## How it works

| Step | What runs |
|---|---|
| Speech-to-text | OpenAI `gpt-4o-transcribe`, Groq Whisper, or browser speech recognition, selected by configuration. |
| Claim extraction and verdicts | Nemotron via Nebius Token Factory, or OpenAI, selected by configuration. |
| Web research | MiniMax-M3 via Nebius, or OpenAI, through the deepagents framework. |
| Sources | Tavily search. Article titles and URLs come from the search results. |

Verdicts are **Supported**, **Contradicted**, **Misleading**, or **Insufficient evidence**. Checks that are still running show as pending. Failed checks stay separate and are not treated as finished verdicts.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A browser with microphone access. Chrome or Edge is required only for browser speech recognition.
- A [Nebius Token Factory](https://tokenfactory.nebius.com/) API key, or an OpenAI API key
- A [Tavily](https://tavily.com/) API key
- Optional: a [Groq](https://console.groq.com/) API key for server-side transcription

## Setup

From the repository root:

```bash
cp env.example .env
```

Edit `.env`:

```env
TAVILY_API_KEY=your-tavily-api-key

# Default: Nebius for LLM work; OpenAI, Groq, or browser for speech-to-text.
AI_PROVIDER=nebius
NEBIUS_API_KEY=your-nebius-api-key

# Optional: use OpenAI for claim extraction, verdicts, research, and STT.
# AI_PROVIDER=openai
# OPENAI_API_KEY=your-openai-api-key
# STT_PROVIDER=openai

# Optional: use Groq Whisper for STT.
# GROQ_API_KEY=your-groq-api-key
# STT_PROVIDER=groq
```

Install dependencies:

```bash
uv sync
```

API keys stay on the server. Do not commit `.env`.

`STT_PROVIDER=auto` is the default: it chooses OpenAI when `OPENAI_API_KEY` is
set, then Groq, then browser speech. The extractor admits only public,
verifiable factual claims; greetings, preferences, plans, questions, and casual
conversation are ignored before research begins.

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
