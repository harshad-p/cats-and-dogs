"""Live Fact Checker — Flask backend.

Architecture
------------
Browser (Web Speech API / Groq Whisper)
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

import ast
import hashlib
import io
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Generator
from urllib.parse import urlparse

import openai
from deepagents import create_deep_agent
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from langchain_core.messages import ToolMessage
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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
            sources.append({
                "title": title,
                "url": url,
                "domain": _domain(url),
                "supports_claim": None,
            })

    # Bare URLs not caught above
    bare_pattern = re.compile(r"https?://\S+")
    for m in bare_pattern.finditer(research_text):
        url = m.group(0).rstrip(").,")
        if url not in seen:
            seen.add(url)
            sources.append({
                "title": _domain(url),
                "url": url,
                "domain": _domain(url),
                "supports_claim": None,
            })

    return sources[:10]  # cap at 10


def _domain(url: str) -> str:
    """Return the publisher host, without a leading www."""
    try:
        host = urlparse(url).netloc or url
    except Exception:
        return url
    if host.lower().startswith("www."):
        return host[4:]
    return host


def _is_site_root(url: str) -> bool:
    """True when a URL is a publisher homepage rather than a specific article.

    Only the site root (and common index paths) count. A short article path
    is left alone — this does not guess which URL is "the" article.
    """
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    path = (parsed.path or "").rstrip("/").lower()
    return path in ("", "index.html", "index.htm", "index.php", "home", "default.aspx")


def _url_key(url: str) -> str:
    """Identity used to dedupe the same page cited with trivial URL differences."""
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if _is_site_root(url):
        return host
    return f"{host}{(parsed.path or '').rstrip('/')}"


def _host_key(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _parse_tool_payload(content):
    """Parse a Tavily tool result from a string, dict, or content-block list."""
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        if content and all(isinstance(item, dict) and item.get("url") for item in content):
            return {"results": content}
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                texts.append(str(block.get("text") or block.get("content") or ""))
        content = "\n".join(texts)
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"results": parsed}
    start = text.find("{")
    end = text.rfind("}")
    snippet = text[start:end + 1] if start != -1 and end > start else text
    for candidate in (snippet, text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return parsed if isinstance(parsed, dict) else {"results": parsed}
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"results": parsed}
    return None


def _sources_from_tavily_messages(messages: list) -> list[dict]:
    """Extract article URLs from Tavily tool results in the agent history.

    Homepage links come from MiniMax's free-text report, not from Tavily.
    The model often shortens a result such as
    ``https://www.washingtonpost.com/politics/2024/…/story/`` to
    ``https://www.washingtonpost.com/`` when it writes the markdown report.
    ``_sources_from_research()`` can only see that paraphrased text.

    The tool messages still hold Tavily's JSON, including the original
    ``title`` and ``url``. Those values are copied verbatim. Nothing in
    this function builds or guesses a URL.
    """
    sources: list[dict] = []
    seen: set[str] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = (getattr(msg, "name", "") or "").lower()
        if "tavily" not in name:
            continue
        payload = _parse_tool_payload(getattr(msg, "content", None))
        if payload is None:
            payload = _parse_tool_payload(getattr(msg, "artifact", None))
        if not isinstance(payload, dict):
            continue
        results = payload.get("results") or []
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            url = (result.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            key = _url_key(url)
            if key in seen:
                continue
            seen.add(key)
            title = (result.get("title") or "").strip() or _domain(url)
            sources.append({
                "title": title,
                "url": url,
                "domain": _domain(url),
                "from_tavily": True,
                "supports_claim": None,
            })

    return sources


def _present_source(source: dict) -> dict:
    """Shape a source for the UI and the PDF.

    A homepage keeps its search-result title when Tavily supplied it, and is
    flagged so the UI can say "Homepage" instead of presenting it as an article.
    A homepage that only appears in model text is labeled with its domain —
    the model headline is not treated as an article title.
    """
    url = (source.get("url") or "").strip()
    domain = source.get("domain") or _domain(url)
    title = (source.get("title") or "").strip()
    homepage = _is_site_root(url)
    if homepage and not source.get("from_tavily"):
        title = domain or url
    elif not title or title == url:
        title = domain or url
    presented = {
        "title": title,
        "url": url,
        "domain": domain,
        "is_homepage": homepage,
        "supports_claim": source.get("supports_claim") if isinstance(source.get("supports_claim"), bool) else None,
    }
    return presented


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
                    "Only use evidence explicitly mentioned in the report. Do NOT fabricate sources. "
                    "Copy source URLs exactly. Do not invent an article URL, and do not "
                    "replace an article URL with the publisher homepage."
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
    "Cite the exact article URL returned by the search tool. "
    "Do not replace an article URL with the publisher homepage."
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
        all_messages = result["messages"]
        research_text = all_messages[-1].content

        # Extract article URLs verbatim from Tavily ToolMessages — these are
        # the actual search-result URLs before any LLM paraphrasing.
        tavily_sources = _sources_from_tavily_messages(all_messages)
        log.info(
            "research_claim %s: %d Tavily sources extracted from messages",
            claim_id, len(tavily_sources),
        )

        # Emit research text for display
        _update_claim(claim_id, status="analyzing", research=research_text)

        # Nemotron converts the free-text research into a structured verdict
        vd = structured_verdict_nemotron(claim_text, research_text)

        # Merge sources: prefer Tavily article URLs (verbatim) over LLM-parsed
        # URLs.  Nemotron's sources (from tool call path) may have supports_claim
        # annotations; keep those for any URL that Tavily also returned so we
        # preserve the annotation.  Add any Tavily URL not already present.
        merged = _merge_sources(
            vd.get("sources") or [],
            tavily_sources,
            research_text if isinstance(research_text, str) else "",
        )

        _update_claim(
            claim_id,
            status="done",
            verdict=vd["verdict"],
            one_liner=vd["one_liner"],
            explanation=vd["explanation"],
            confidence=vd["confidence"],
            sources=merged,
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


def _merge_sources(
    nemotron_sources: list[dict],
    tavily_sources: list[dict],
    research_text: str = "",
) -> list[dict]:
    """Prefer Tavily article URLs over homepage citations written by the model.

    Tavily titles and URLs are kept verbatim. A model-written URL is added
    only when no Tavily results exist, or when that exact URL appears in the
    research report. A homepage is dropped when the same host already has a
    more specific URL — the homepage is not shown as a second "article".
    No URL is constructed.
    """
    merged: list[dict] = []
    index_by_key: dict[str, int] = {}
    report = research_text or ""

    def _add(raw: dict, *, from_tavily: bool, require_verbatim: bool) -> None:
        if not isinstance(raw, dict):
            return
        url = (raw.get("url") or "").strip().rstrip(").,")
        if not url.startswith(("http://", "https://")):
            return
        key = _url_key(url)
        if key in index_by_key:
            existing = merged[index_by_key[key]]
            if isinstance(raw.get("supports_claim"), bool):
                existing["supports_claim"] = raw["supports_claim"]
            if from_tavily:
                existing["from_tavily"] = True
                existing["url"] = url
                title = (raw.get("title") or "").strip()
                if title:
                    existing["title"] = title
                existing["domain"] = _domain(url)
            return
        if require_verbatim and url not in report:
            return
        entry = {
            "title": (raw.get("title") or "").strip(),
            "url": url,
            "domain": raw.get("domain") or _domain(url),
            "from_tavily": from_tavily or bool(raw.get("from_tavily")),
            "supports_claim": raw.get("supports_claim") if isinstance(raw.get("supports_claim"), bool) else None,
        }
        index_by_key[key] = len(merged)
        merged.append(entry)

    for source in tavily_sources or []:
        _add(source, from_tavily=True, require_verbatim=False)
    # When Tavily returned results, ignore model URLs that are not verbatim
    # in the report. Those are the paraphrased homepage links.
    for source in nemotron_sources or []:
        _add(source, from_tavily=False, require_verbatim=bool(tavily_sources))

    article_hosts = {
        _host_key(source["url"])
        for source in merged
        if not _is_site_root(source["url"])
    }
    presented: list[dict] = []
    for source in merged:
        if _is_site_root(source["url"]) and _host_key(source["url"]) in article_hosts:
            continue
        presented.append(_present_source(source))
        if len(presented) >= 10:
            break
    return presented


def _verdict_counts(claims_snapshot: list[dict]) -> dict[str, int]:
    """Count each claim once. Superseded claims are excluded.

    Failed checks stay in FAILED even though the pipeline stores
    INSUFFICIENT_EVIDENCE on the error record. Pending checks are not
    given a verdict.
    """
    counts = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "MISLEADING": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "PENDING": 0,
        "FAILED": 0,
    }
    for claim in claims_snapshot:
        status = claim.get("status")
        if status == "done":
            verdict = claim.get("verdict")
            if verdict not in _VERDICT_ENUM:
                verdict = "INSUFFICIENT_EVIDENCE"
            counts[verdict] += 1
        elif status == "error":
            counts["FAILED"] += 1
        elif status == "superseded":
            continue
        else:
            counts["PENDING"] += 1
    return counts


# ── PDF report generation ─────────────────────────────────────────────────────

_VERDICT_COLOUR = {
    "SUPPORTED":            colors.HexColor("#3fb068"),
    "CONTRADICTED":         colors.HexColor("#e8524a"),
    "MISLEADING":           colors.HexColor("#d4a72c"),
    "INSUFFICIENT_EVIDENCE": colors.HexColor("#636b82"),
}
_VERDICT_FRIENDLY = {
    "SUPPORTED":            "Supported",
    "CONTRADICTED":         "Contradicted",
    "MISLEADING":           "Misleading",
    "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
}
_FAILED_COLOUR = colors.HexColor("#e07830")
_PENDING_COLOUR = colors.HexColor("#4d8ef0")
_SUPERSEDED_COLOUR = colors.HexColor("#9e7fe8")


def _pdf_text(value) -> str:
    """Escape text for a ReportLab paragraph. Newlines become line breaks."""
    text = "" if value is None else str(value)
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return text.replace("\n", "<br/>")


def _pdf_attr(value: str) -> str:
    return (
        (value or "")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "")
        .replace(">", "")
    )

def _build_pdf(claims_snapshot: list[dict]) -> bytes:
    """Render a ReportLab PDF from the current claims state.

    Returns raw PDF bytes.  Never includes API keys or secrets.
    All source URLs are taken verbatim from the claims data —
    nothing is fabricated.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.2 * cm,
        title="Live Fact Checker — Report",
        author="Live Fact Checker",
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 4.4 * cm   # usable text width

    # Custom styles
    sTitle = ParagraphStyle(
        "sTitle",
        parent=styles["Heading1"],
        fontSize=20, leading=26, spaceAfter=4,
        textColor=colors.HexColor("#1f2328"),
    )
    sMeta = ParagraphStyle(
        "sMeta",
        parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#57606a"),
        spaceAfter=16,
    )
    sH2 = ParagraphStyle(
        "sH2",
        parent=styles["Heading2"],
        fontSize=13, leading=18, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1f2328"),
    )
    sBodyText = ParagraphStyle(
        "sBody",
        parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=6,
        textColor=colors.HexColor("#1f2328"),
    )
    sMuted = ParagraphStyle(
        "sMuted",
        parent=styles["Normal"],
        fontSize=9, leading=13, spaceAfter=4,
        textColor=colors.HexColor("#57606a"),
    )
    sClaimText = ParagraphStyle(
        "sClaimText",
        parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=4,
        textColor=colors.HexColor("#1f2328"),
        fontName="Helvetica-Bold",
    )
    sSourceLink = ParagraphStyle(
        "sSourceLink",
        parent=styles["Normal"],
        fontSize=9, leading=13,
        textColor=colors.HexColor("#3b82d4"),
    )

    def page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#57606a"))
        canvas.drawString(2.2 * cm, 1.2 * cm, "Live Fact Checker")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    story = []

    # ── Title block ────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%d %B %Y %H:%M UTC")
    story.append(Paragraph("Fact-Check Report", sTitle))
    story.append(Paragraph(f"Generated {ts}", sMeta))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=12))

    # ── Summary table ──────────────────────────────────────────────────────────
    counts = _verdict_counts(claims_snapshot)

    story.append(Paragraph("Summary", sH2))
    story.append(Paragraph(
        "Completed checks are counted once, by verdict. "
        "Pending and failed checks are listed separately and are not treated as finished verdicts. "
        "Superseded claims are omitted from these totals.",
        sMuted,
    ))
    summary_data = [["Verdict", "Count"]]
    for v_key in ("SUPPORTED", "CONTRADICTED", "MISLEADING", "INSUFFICIENT_EVIDENCE"):
        summary_data.append([_VERDICT_FRIENDLY[v_key], str(counts[v_key])])
    summary_data.append(["Pending", str(counts["PENDING"])])
    summary_data.append(["Failed", str(counts["FAILED"])])

    tbl = Table(summary_data, colWidths=[W * 0.7, W * 0.3])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#f7f8fa")),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#1f2328")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ── Per-claim sections ─────────────────────────────────────────────────────
    story.append(Paragraph("Claims", sH2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb"), spaceAfter=8))

    pill_width = 4.8 * cm
    for idx, c in enumerate(claims_snapshot, 1):
        verdict = c.get("verdict") or ""
        status = c.get("status") or "queued"
        if status == "error":
            v_label, v_colour = "Failed", _FAILED_COLOUR
        elif status in ("queued", "researching", "analyzing"):
            v_label, v_colour = "Pending", _PENDING_COLOUR
        elif status == "superseded":
            v_label, v_colour = "Superseded", _SUPERSEDED_COLOUR
        else:
            v_label = _VERDICT_FRIENDLY.get(verdict, "Insufficient evidence")
            v_colour = _VERDICT_COLOUR.get(verdict, colors.HexColor("#636b82"))

        pill = Table([[v_label]], colWidths=[pill_width])
        pill.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), v_colour),
            ("TEXTCOLOR",    (0, 0), (-1, -1), colors.white),
            ("FONTNAME",     (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        header_row = Table(
            [[Paragraph(f"Claim {idx}", sMuted), pill]],
            colWidths=[W - pill_width - 0.4 * cm, pill_width],
        )
        header_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))

        block: list = [header_row, Spacer(1, 0.15 * cm)]
        spoken = (c.get("sentence") or "").strip()
        claim_text = (c.get("text") or "").strip()
        block.append(Paragraph("Spoken sentence", sMuted))
        block.append(Paragraph(
            _pdf_text(spoken) if spoken else "Not stored for this claim.",
            sBodyText,
        ))
        block.append(Paragraph("Extracted claim", sMuted))
        block.append(Paragraph(_pdf_text(claim_text) or "—", sClaimText))

        if status in ("queued", "researching", "analyzing"):
            block.append(Paragraph(
                "Research in progress. This check is not completed.",
                sMuted,
            ))
        elif status == "error":
            reason = (c.get("explanation") or "The research request failed.")[:400]
            block.append(Paragraph(
                "Research failed. This check is not completed. "
                + _pdf_text(reason),
                sMuted,
            ))
        elif status == "superseded":
            previous = _VERDICT_FRIENDLY.get(verdict)
            note = "Superseded by a later correction. Excluded from the verdict totals."
            if previous:
                note += f" Earlier verdict: {previous}."
            block.append(Paragraph(note, sMuted))
        else:
            if c.get("one_liner"):
                block.append(Paragraph(_pdf_text(c["one_liner"]), sBodyText))
            if c.get("explanation"):
                block.append(Paragraph("Explanation", sMuted))
                block.append(Paragraph(_pdf_text(c["explanation"]), sBodyText))
            conf = c.get("confidence")
            if isinstance(conf, (int, float)):
                block.append(Paragraph(f"Confidence: {int(conf * 100)}%", sMuted))
            sources = c.get("sources") or []
            block.append(Paragraph("Sources", sMuted))
            if not sources:
                block.append(Paragraph("No sources were returned for this check.", sMuted))
            for source in sources:
                url = (source.get("url") or "").strip()
                title = (source.get("title") or url).strip()
                domain = source.get("domain") or (_domain(url) if url else "")
                supports = source.get("supports_claim")
                if supports is True:
                    relation = "Supports the claim. "
                elif supports is False:
                    relation = "Disputes the claim. "
                else:
                    relation = ""
                kind = "Homepage" if source.get("is_homepage") else (domain or "Source")
                if url:
                    block.append(Paragraph(
                        f'{relation}<link href="{_pdf_attr(url)}">{_pdf_text(title)}</link>'
                        f' <font color="#57606a">({_pdf_text(kind)})</font>',
                        sSourceLink,
                    ))
                else:
                    block.append(Paragraph(_pdf_text(title), sMuted))

        block.append(Spacer(1, 0.25 * cm))
        block.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#e5e7eb"), spaceAfter=8,
        ))
        story.append(KeepTogether(block))

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return buf.getvalue()


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
                "sentence": transcript,
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


@app.route("/api/report", methods=["GET"])
def api_report():
    """Generate and return a PDF fact-check report from current session claims."""
    with claims_lock:
        snapshot = list(claims.values())

    if not snapshot:
        return jsonify({"error": "No claims in this session"}), 400

    try:
        pdf_bytes = _build_pdf(snapshot)
    except Exception as exc:
        log.exception("PDF generation failed")
        return jsonify({"error": str(exc)}), 500

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    filename = f"fact-check-report-{ts}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    log.info("Starting Live Fact Checker on http://localhost:%d", port)
    app.run(debug=False, port=port, threaded=True)
