# Live Fact Checker

Real-time speech-to-text fact-checking powered by **Nebius Token Factory**, **NVIDIA Nemotron**, **MiniMax-M3**, and **Tavily**.

## Architecture

```
Browser (Web Speech API / Groq Whisper)
  └─► POST /api/transcript        raw transcript text
        └─► Nemotron-3.5-Lightning  structured claim extraction  (tool call)
              └─► for each new/corrected claim
                    └─► SSE  /api/check/<id>
                          ├─► MiniMax-M3 + deepagents loop  web research via Tavily
                          └─► Nemotron-3.5-Lightning         structured verdict (tool call)
```

| Role | Model | Why |
|---|---|---|
| Claim extraction & verdict structuring | `nvidia/Nemotron-3_5-Lightning` | Fast, structured tool-calling; Nemotron side-challenge requirement |
| Web research agent | `MiniMaxAI/MiniMax-M3` | Proven with Tavily in the existing baseline |
| Web search | Tavily `search_depth=advanced` | Real sources, no fabrication |
| Speech-to-text (primary) | Browser Web Speech API | Zero extra credentials; works in Chrome/Edge |
| Speech-to-text (upgrade) | Groq `whisper-large-v3` | Fast cloud STT; requires `GROQ_API_KEY` |

---

## Setup

### 1 — Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (already used in this project)
- PortAudio (macOS: `brew install portaudio`)
- Chrome or Edge (for Web Speech API)

### 2 — Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp env.example .env
```

```env
# Required
NEBIUS_API_KEY=v1.CmQK...          # Nebius Token Factory API key
TAVILY_API_KEY=tvly-dev-...        # Tavily web search API key

# Optional — enables server-side Whisper via Groq (faster, no browser needed)
GROQ_API_KEY=gsk_...               # https://console.groq.com → free tier
```

### 3 — Install dependencies

```bash
cd agents/langchain/deep-agent-example-1
uv sync
```

---

## Launch

```bash
cd agents/langchain/deep-agent-example-1
.venv/bin/python app.py
```

Open **http://localhost:5050** in Chrome or Edge.

To use a different port:

```bash
PORT=8080 .venv/bin/python app.py
```

---

## Two-Minute Demo Script

1. Open **http://localhost:5050** in Chrome.
2. Click **🎙 Start Listening**.
3. Speak:
   > *"The Great Wall of China is visible from the Moon with the naked eye."*
4. Pause — watch the transcript commit and a **CONTRADICTED** card appear in the right panel.
   Tavily searches fire and sources are cited.
5. Speak a correction:
   > *"Actually, the Great Wall is only visible from low Earth orbit with a zoom lens."*
6. The first card is marked **Superseded**; a new **MISLEADING** card appears.
7. Speak a true claim:
   > *"The Eiffel Tower is 330 metres tall including the antenna."*
8. A **SUPPORTED** card appears with source citations.
9. Click **Reset** to clear the session.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/transcript` | Submit transcript text; returns extracted claims |
| `GET` | `/api/claims` | All current claims |
| `GET` | `/api/check/<id>` | SSE stream of updates for one claim |
| `POST` | `/api/stt` | Server-side STT (requires `GROQ_API_KEY`) |
| `POST` | `/api/reset` | Clear all claims |

---

## Adding Groq Whisper (upgrade path)

1. Sign up at [console.groq.com](https://console.groq.com) (free tier available).
2. Create an API key.
3. Add `GROQ_API_KEY=gsk_...` to `.env`.
4. Restart the server.

The frontend will automatically POST audio blobs to `/api/stt` when Groq is configured,
enabling server-side transcription independent of the browser's Speech API.

---

## Claim extraction examples

| Input | Extracted claims |
|---|---|
| *"Paris is the capital of France and the Eiffel Tower is 330m tall."* | 2 claims |
| *"Wait, I meant the tower is 300m."* | Corrects previous height claim |
| *"I think the food is great."* | 0 claims (opinion filtered) |

---

## Security notes

- API keys are read from `.env` on the server. They are **never** sent to the browser.
- The `.env` file is listed in `.gitignore`.
- Do not commit `.env` files.
