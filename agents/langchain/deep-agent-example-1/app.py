"""Live Fact Checker — Flask backend.

Architecture
------------
Browser (Web Speech API)
  └─► POST /api/transcript          raw transcript text
        └─► Nemotron-3_5-Lightning  structured claim extraction (tool call)
              └─► for each new/corrected claim
                    └─► SSE stream /api/check/<claim_id>
                          └─► MiniMax-M3 deep agent + Tavily   research + verdict
                                └─► Nemotron-3_5-Lightning      final structured verdict

Environment
-----------
NEBIUS_API_KEY   required — Nebius Token Factory
TAVILY_API_KEY   required — Tavily web search
GROQ_API_KEY     optional — enables server-side Whisper via Groq (fast cloud STT)
"""

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from typing import Generator

import openai
from deepagents import create_deep_agent
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── API clients ─────────────────────────────────────────────────────────────
NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # optional

nebius_client = openai.OpenAI(
    api_key=NEBIUS_API_KEY,
    base_url="https://api.studio.nebius.ai/v1/",
)

# Optional Groq client for server-side Whisper STT
groq_client = None
if GROQ_API_KEY:
    groq_client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1/",
    )
    log.info("Groq STT enabled (Whisper-large-v3)")
else:
    log.info("No GROQ_API_KEY — browser Web Speech API is the STT provider")

# ── Nemotron tool schemas ────────────────────────────────────────────────────
EXTRACT_CLAIMS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_claims",
        "description": (
            "Extract every distinct, independently verifiable factual claim from the transcript. "
            "Skip personal opinions or questions. "
            "If a later sentence corrects or amends an earlier claim, return the corrected version only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "stable slug, e.g. 'eiffel-tower-location'"},
                            "text": {"type": "string", "description": "the claim verbatim or lightly cleaned"},
                            "corrects": {"type": "string", "description": "id of a previous claim this replaces, if any"},
                        },
                        "required": ["id", "text"],
                    },
                }
            },
            "required": ["claims"],
        },
    },
}

VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "verdict",
        "description": "Return a structured fact-check verdict based on evidence already gathered.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["SUPPORTED", "CONTRADICTED", "MISLEADING", "INSUFFICIENT_EVIDENCE"],
                },
                "one_liner": {"type": "string", "description": "≤15-word summary for the UI badge"},
                "explanation": {"type": "string", "description": "2–4 sentence explanation citing the evidence"},
                "confidence": {"type": "number", "description": "0.0–1.0"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "supports_claim": {"type": "boolean"},
                        },
                        "required": ["title", "url", "supports_claim"],
                    },
                },
            },
            "required": ["verdict", "one_liner", "explanation", "confidence", "sources"],
        },
    },
}

# ── In-memory state ──────────────────────────────────────────────────────────
# { claim_id: { text, status, verdict, sources, ... } }
claims: dict[str, dict] = {}
claims_lock = threading.Lock()

# SSE queues: { claim_id: [queue, ...] }
sse_queues: dict[str, list[queue.Queue]] = {}
sse_lock = threading.Lock()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _claim_slug(text: str) -> str:
    """Deterministic slug from claim text (first 6 chars of sha1)."""
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:8]


def _push_sse(claim_id: str, event: str, data: dict) -> None:
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        for q in sse_queues.get(claim_id, []):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


def _update_claim(claim_id: str, **kwargs) -> None:
    with claims_lock:
        claims[claim_id].update(kwargs)
    _push_sse(claim_id, "update", {"claim_id": claim_id, **kwargs})


# ── Nemotron: claim extraction ────────────────────────────────────────────────

def extract_claims_nemotron(transcript: str) -> list[dict]:
    """Use Nemotron-3_5-Lightning to extract structured claims from transcript."""
    resp = nebius_client.chat.completions.create(
        model="nvidia/Nemotron-3_5-Lightning",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise claim extractor. "
                    "Extract factual claims only. Ignore rhetorical questions and opinions."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        tools=[EXTRACT_CLAIMS_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_claims"}},
        max_tokens=800,
        temperature=0.0,
    )
    choice = resp.choices[0]
    if choice.message.tool_calls:
        raw = json.loads(choice.message.tool_calls[0].function.arguments)
        return raw.get("claims", [])
    return []


# ── Verdict normalisation ─────────────────────────────────────────────────────

# MiniMax-M3 uses TRUE/FALSE in its reports; map them to the UI's verdict enum.
_VERDICT_MAP: dict[str, str] = {
    "TRUE":                  "SUPPORTED",
    "FALSE":                 "CONTRADICTED",
    "SUPPORTED":             "SUPPORTED",
    "CONTRADICTED":          "CONTRADICTED",
    "MISLEADING":            "MISLEADING",
    "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    "INSUFFICIENT EVIDENCE": "INSUFFICIENT_EVIDENCE",
}

_VERDICT_ENUM = frozenset(_VERDICT_MAP.values())


def _normalise_verdict(raw: str) -> str | None:
    """Map a raw verdict string to one of the four UI enum values, or None."""
    return _VERDICT_MAP.get(raw.strip().upper().replace("-", "_"))


def _parse_xml_tool_content(content: str) -> dict:
    """Extract fields from Nemotron's <parameter=FIELD>VALUE</parameter> format.

    When Nemotron-3_5-Lightning is truncated (finish_reason=length) it falls
    back from OpenAI-format tool calls to inline XML in the text content:

        <tool_call>
        <function=verdict>
        <parameter=verdict>
        CONTRADICTED
        </parameter>
        <parameter=one_liner>…</parameter>
        …

    This parser recovers every field it can find; missing fields are left as
    None so callers can decide how to fill them in.
    """
    def _get(field: str) -> str | None:
        m = re.search(
            rf"<parameter={re.escape(field)}>\s*(.*?)\s*(?:</parameter>|<parameter=|\Z)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else None

    verdict_raw = _get("verdict")
    verdict = _normalise_verdict(verdict_raw) if verdict_raw else None

    one_liner = _get("one_liner")
    explanation = _get("explanation")

    confidence: float | None = None
    conf_raw = _get("confidence")
    if conf_raw:
        try:
            confidence = float(conf_raw)
        except ValueError:
            pass

    sources: list[dict] = []
    src_raw = _get("sources")
    if src_raw:
        # The sources value may be a JSON array (possibly truncated)
        # Try to parse it; fall back to an empty list rather than crashing.
        try:
            sources = json.loads(src_raw)
        except json.JSONDecodeError:
            # Attempt to salvage a truncated JSON array
            trimmed = src_raw.rstrip(", \n")
            try:
                sources = json.loads(trimmed + "]")
            except json.JSONDecodeError:
                sources = []

    return {
        "verdict": verdict,
        "one_liner": one_liner,
        "explanation": explanation,
        "confidence": confidence,
        "sources": sources,
    }


def _extract_verdict_from_research(research_text: str) -> str | None:
    """Last-resort: read the verdict that MiniMax wrote in plain English.

    Scans for patterns like '**Verdict: FALSE**', '# Verdict: CONTRADICTED',
    'Verdict: TRUE', etc. Only returns a value that maps to a known enum entry.
    Never fabricates — if nothing recognisable is found, returns None.
    """
    pattern = re.compile(
        r"(?:verdict|result)[:\s*]+([A-Z_/ ]+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(research_text):
        candidate = m.group(1).strip().rstrip("*").strip()
        # Take only the first word so "FALSE — The claim…" → "FALSE"
        first_word = candidate.split()[0] if candidate else ""
        mapped = _normalise_verdict(first_word)
        if mapped:
            return mapped
    return None


def _sources_from_research(research_text: str) -> list[dict]:
    """Extract source URLs mentioned in the research report.

    Finds Markdown-style '- Title: URL' or bare URLs.  Never fabricates —
    only returns URLs that appear verbatim in the text.
    """
    sources: list[dict] = []
    seen: set[str] = set()

    # Pattern: '- Title — *optional italics*: URL' or '- Title: URL'
    md_pattern = re.compile(
        r"-\s+([^:\n]+?)(?:\s+—[^:]+)?:\s+(https?://\S+)",
        re.MULTILINE,
    )
    for m in md_pattern.finditer(research_text):
        title = m.group(1).strip().strip("*")
        url = m.group(2).rstrip(").,")
        if url not in seen:
            seen.add(url)
            sources.append({"title": title, "url": url, "supports_claim": False})

    # Bare URLs not caught above
    bare_pattern = re.compile(r"https?://\S+")
    for m in bare_pattern.finditer(research_text):
        url = m.group(0).rstrip(").,")
        if url not in seen:
            seen.add(url)
            sources.append({"title": url, "url": url, "supports_claim": False})

    return sources[:10]  # cap at 10


# ── Nemotron: structured verdict from research text ───────────────────────────

def structured_verdict_nemotron(claim_text: str, research_text: str) -> dict:
    """Use Nemotron to parse the MiniMax research report into a structured verdict.

    Failure modes handled:
    1. Normal path  — finish_reason=tool_calls, tool_calls populated → parse JSON args.
    2. Truncation   — finish_reason=length, content contains <parameter=…> XML →
                      parse XML with _parse_xml_tool_content().
    3. No tool call — model responded in plain text → fall back to
                      _extract_verdict_from_research() on the research text itself.
    4. Truly unknown → return INSUFFICIENT_EVIDENCE (never a fabricated verdict).
    """
    resp = nebius_client.chat.completions.create(
        model="nvidia/Nemotron-3_5-Lightning",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-checking analyst. "
                    "Given a claim and a research report, extract a structured verdict. "
                    "Only use evidence explicitly mentioned in the report. Do NOT fabricate sources."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CLAIM: {claim_text}\n\n"
                    f"RESEARCH REPORT:\n{research_text}"
                ),
            },
        ],
        tools=[VERDICT_TOOL],
        tool_choice={"type": "function", "function": {"name": "verdict"}},
        max_tokens=1500,   # raised from 1000 — full response needs ~600–800 tokens
        temperature=0.0,
    )
    choice = resp.choices[0]

    # ── Path 1: clean structured tool call ───────────────────────────────────
    if choice.message.tool_calls:
        parsed = json.loads(choice.message.tool_calls[0].function.arguments)
        # Normalise the verdict in case the model used TRUE/FALSE
        parsed["verdict"] = _normalise_verdict(parsed.get("verdict", "")) or "INSUFFICIENT_EVIDENCE"
        return parsed

    # ── Path 2: truncated XML in content field ────────────────────────────────
    content = choice.message.content or ""
    if "<parameter=" in content:
        log.warning(
            "structured_verdict_nemotron: finish_reason=%s, falling back to XML parser",
            choice.finish_reason,
        )
        xml_result = _parse_xml_tool_content(content)
        if xml_result.get("verdict"):
            # Fill in any gaps from the research text
            if not xml_result["sources"]:
                xml_result["sources"] = _sources_from_research(research_text)
            if not xml_result["explanation"]:
                xml_result["explanation"] = research_text[:400]
            if xml_result["confidence"] is None:
                xml_result["confidence"] = 0.8
            if not xml_result["one_liner"]:
                xml_result["one_liner"] = xml_result["verdict"].replace("_", " ").title()
            return xml_result

    # ── Path 3: no tool call at all — read verdict from the research text ─────
    log.warning(
        "structured_verdict_nemotron: no tool call and no XML (finish_reason=%s); "
        "extracting verdict from research text",
        choice.finish_reason,
    )
    fallback_verdict = _extract_verdict_from_research(research_text) or "INSUFFICIENT_EVIDENCE"
    return {
        "verdict": fallback_verdict,
        "one_liner": fallback_verdict.replace("_", " ").title(),
        "explanation": research_text[:400],
        "confidence": 0.5,
        "sources": _sources_from_research(research_text),
    }


# ── MiniMax deep agent: Tavily research ───────────────────────────────────────

FACT_CHECK_SYSTEM_PROMPT = (
    "You are a fact-checking assistant. "
    "You MUST call the tavily_search tool at least once before writing any conclusion. "
    "Never rely solely on your training knowledge to verify a claim. "
    "Always cite the URLs of your sources."
)


def research_claim(claim_id: str, claim_text: str) -> None:
    """Run deep agent research in a background thread, push SSE updates."""
    _update_claim(claim_id, status="researching")

    try:
        tavily_search = TavilySearch(max_results=5, search_depth="advanced")
        model = ChatNebius(model="MiniMaxAI/MiniMax-M3")
        agent = create_deep_agent(
            model=model,
            tools=[tavily_search],
            system_prompt=FACT_CHECK_SYSTEM_PROMPT,
        )

        question = (
            f"Verify this claim: '{claim_text}' "
            "Find reliable sources, determine whether the evidence supports "
            "or contradicts the claim, and cite the sources. "
            "If the evidence is insufficient, say so."
        )

        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        research_text = result["messages"][-1].content

        # Emit research text for display
        _update_claim(claim_id, status="analyzing", research=research_text)

        # Nemotron converts the free-text research into a structured verdict
        vd = structured_verdict_nemotron(claim_text, research_text)
        _update_claim(
            claim_id,
            status="done",
            verdict=vd["verdict"],
            one_liner=vd["one_liner"],
            explanation=vd["explanation"],
            confidence=vd["confidence"],
            sources=vd["sources"],
        )

    except Exception as exc:
        log.exception("research_claim failed for %s", claim_id)
        _update_claim(
            claim_id,
            status="error",
            verdict="INSUFFICIENT_EVIDENCE",
            one_liner="Research error",
            explanation=str(exc),
            confidence=0.0,
            sources=[],
        )
    finally:
        # Signal stream end
        _push_sse(claim_id, "done", {"claim_id": claim_id})


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcript", methods=["POST"])
def api_transcript():
    """Receive transcript text, extract claims, start background research."""
    body = request.get_json(force=True)
    transcript = (body.get("text") or "").strip()
    if not transcript:
        return jsonify({"error": "empty transcript"}), 400

    # Nemotron extracts claims
    extracted = extract_claims_nemotron(transcript)
    new_claims = []

    with claims_lock:
        for c in extracted:
            cid = c.get("id") or _claim_slug(c["text"])
            corrects = c.get("corrects")

            # If this corrects an old claim, mark old one superseded
            if corrects and corrects in claims:
                claims[corrects]["status"] = "superseded"
                _push_sse(corrects, "update", {"claim_id": corrects, "status": "superseded"})

            # Skip if identical claim already in a terminal state
            existing = claims.get(cid)
            if existing and existing["status"] in ("done", "researching", "analyzing"):
                continue

            claims[cid] = {
                "id": cid,
                "text": c["text"],
                "status": "queued",
                "verdict": None,
                "one_liner": None,
                "explanation": None,
                "confidence": None,
                "sources": [],
                "research": None,
                "corrects": corrects,
            }
            new_claims.append(cid)

    # Fire background research for each new claim
    for cid in new_claims:
        t = threading.Thread(target=research_claim, args=(cid, claims[cid]["text"]), daemon=True)
        t.start()

    return jsonify({
        "extracted": extracted,
        "queued": new_claims,
        "all_claims": list(claims.keys()),
    })


@app.route("/api/claims", methods=["GET"])
def api_claims():
    with claims_lock:
        return jsonify(list(claims.values()))


@app.route("/api/check/<claim_id>")
def api_check_stream(claim_id: str):
    """SSE stream of updates for a single claim."""
    q: queue.Queue = queue.Queue(maxsize=50)
    with sse_lock:
        sse_queues.setdefault(claim_id, []).append(q)

    # If claim is already done, send current state immediately and close
    with claims_lock:
        current = claims.get(claim_id)

    def generate() -> Generator[str, None, None]:
        if current:
            yield f"event: update\ndata: {json.dumps({'claim_id': claim_id, **current})}\n\n"
            if current.get("status") in ("done", "error", "superseded"):
                yield f"event: done\ndata: {json.dumps({'claim_id': claim_id})}\n\n"
                return

        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                    if '"event": "done"' in msg or 'event: done' in msg:
                        break
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with sse_lock:
                lst = sse_queues.get(claim_id, [])
                if q in lst:
                    lst.remove(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/stt", methods=["POST"])
def api_stt():
    """Optional server-side STT via Groq Whisper.
    POST multipart/form-data with field 'audio' (webm/mp4/wav blob).
    Falls back to 400 if GROQ_API_KEY is not set.
    """
    if not groq_client:
        return jsonify({"error": "No GROQ_API_KEY configured. Use browser Web Speech API."}), 400

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file"}), 400

    audio_bytes = audio_file.read()
    lang = request.form.get("lang", "en")
    log.info("STT request: %d bytes, lang=%s", len(audio_bytes), lang)

    try:
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("audio.webm", audio_bytes, "audio/webm"),
            language=lang,
            response_format="text",
        )
        return jsonify({"text": transcript})
    except Exception as exc:
        log.exception("Groq STT failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/config", methods=["GET"])
def api_config():
    """Return frontend-safe configuration (no secrets)."""
    return jsonify({
        "stt_mode": "groq" if groq_client else "browser",
        "groq_model": "whisper-large-v3" if groq_client else None,
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with claims_lock:
        claims.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    log.info("Starting Live Fact Checker on http://localhost:%d", port)
    app.run(debug=False, port=port, threaded=True)
