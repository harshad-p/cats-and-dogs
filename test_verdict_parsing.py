"""Regression tests for structured_verdict_nemotron parsing.

Covers the three failure modes identified in the original bug:
  1. Normal path  — tool_calls populated with JSON args
  2. Truncation   — finish_reason=length, content contains <parameter=…> XML
  3. No tool call — plain text response, verdict read from research text
  4. _normalise_verdict handles TRUE/FALSE from MiniMax reports
  5. _extract_verdict_from_research handles the exact failing report format
  6. _sources_from_research extracts URLs without fabrication
"""

import json
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Minimal stubs so app.py can be imported without live API keys ─────────────
import os
os.environ.setdefault("NEBIUS_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")


def _make_fake_modules():
    """Stub out heavy imports that are not needed for unit tests."""
    for name in [
        "deepagents",
        "langchain_nebius",
        "langchain_tavily",
        "dotenv",
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod

    # dotenv.load_dotenv must be callable
    sys.modules["dotenv"].load_dotenv = lambda: None  # type: ignore[attr-defined]

    # langchain_nebius.ChatNebius
    sys.modules["langchain_nebius"].ChatNebius = MagicMock  # type: ignore[attr-defined]

    # langchain_tavily.TavilySearch
    sys.modules["langchain_tavily"].TavilySearch = MagicMock  # type: ignore[attr-defined]

    # deepagents.create_deep_agent
    sys.modules["deepagents"].create_deep_agent = MagicMock  # type: ignore[attr-defined]

    # flask stub — Flask() must return an object whose .route() works as a decorator
    def _noop_decorator(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    flask_instance = MagicMock()
    flask_instance.route = _noop_decorator

    flask_cls = MagicMock(return_value=flask_instance)

    flask_mod = types.ModuleType("flask")
    flask_mod.Flask = flask_cls  # type: ignore[attr-defined]
    flask_mod.Response = MagicMock  # type: ignore[attr-defined]
    flask_mod.jsonify = MagicMock  # type: ignore[attr-defined]
    flask_mod.render_template = MagicMock  # type: ignore[attr-defined]
    flask_mod.request = MagicMock  # type: ignore[attr-defined]
    flask_mod.send_file = MagicMock  # type: ignore[attr-defined]
    flask_mod.stream_with_context = lambda g: g  # type: ignore[attr-defined]
    sys.modules["flask"] = flask_mod

    # openai stub — nebius_client is constructed at module level
    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = MagicMock  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_mod


_make_fake_modules()

# Now import the helpers we're testing directly from the module
import importlib
import importlib.util, pathlib

_spec = importlib.util.spec_from_file_location(
    "app",
    pathlib.Path(__file__).parent / "app.py",
)
_app = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_app)  # type: ignore[union-attr]

_normalise_verdict             = _app._normalise_verdict
_parse_xml_tool_content        = _app._parse_xml_tool_content
_extract_verdict_from_research = _app._extract_verdict_from_research
_sources_from_research         = _app._sources_from_research
_sources_from_tavily_messages  = _app._sources_from_tavily_messages
_merge_sources                 = _app._merge_sources
_domain                        = _app._domain
_verdict_counts                = _app._verdict_counts
_build_pdf                     = _app._build_pdf
structured_verdict_nemotron    = _app.structured_verdict_nemotron
filter_verifiable_claims       = _app.filter_verifiable_claims
_normalise_check_context       = _app._normalise_check_context
_context_summary               = _app._context_summary
_speaker_identity_conflict     = _app._speaker_identity_conflict

# ── Fixtures ──────────────────────────────────────────────────────────────────

# The exact report that triggered the original bug report.
EARTH_FLAT_REPORT = '''\
# Fact-Check Result: "The Earth is flat."

**Verdict: FALSE** — The claim is contradicted by overwhelming scientific evidence.

## Evidence That Contradicts the Claim

The Earth is not flat. It is an **oblate spheroid** — approximately spherical,
slightly flattened at the poles and bulging at the equator due to its rotation.

## Sources

- NASA — *90 Years of Our Changing Views of Earth*: https://www.nasa.gov/history/90-years-of-our-changing-views-of-earth
- Wikipedia — *Empirical evidence for the spherical shape of Earth*: https://en.wikipedia.org/wiki/Empirical_evidence_for_the_spherical_shape_of_Earth
- Wikipedia — *Spherical Earth*: https://en.wikipedia.org/wiki/Spherical_Earth
- Space.com — *How to debate a flat-Earther*: https://www.space.com/how-to-debate-flat-earther.html

## Conclusion

There is no credible scientific evidence supporting a flat Earth.
'''

# Nemotron XML output as produced when finish_reason=length
XML_TRUNCATED_CONTENT = (
    "<tool_call>\n"
    "<function=verdict>\n"
    "<parameter=verdict>\n"
    "CONTRADICTED\n"
    "</parameter>\n"
    "<parameter=one_liner>\n"
    "Earth is an oblate spheroid, not flat.\n"
    "</parameter>\n"
    "<parameter=explanation>\n"
    "Multiple sources confirm the Earth is spherical.\n"
    "</parameter>\n"
    "<parameter=confidence>\n"
    "0.99\n"
    "</parameter>\n"
    "<parameter=sources>\n"
    '[{"title":"NASA","url":"https://nasa.gov","supports_claim":false}]\n'
    "</parameter>\n"
)

# Severely truncated — sources JSON cut mid-array
XML_SEVERELY_TRUNCATED = (
    "<tool_call>\n"
    "<function=verdict>\n"
    "<parameter=verdict>\n"
    "CONTRADICTED\n"
    "</parameter>\n"
    "<parameter=one_liner>\n"
    "The Earth is not flat.\n"
    "</parameter>\n"
    "<parameter=explanation>\n"
    "Evidence shows Earth is spherical.\n"
    "</parameter>\n"
    "<parameter=confidence>\n"
    "0.99\n"
    "</parameter>\n"
    "<parameter=sources>\n"
    '[{"title":"NASA","url":"https://nasa.gov","supports_claim":false},{"title":"Wiki","url":"https://en.wi'
)


def _mock_choice(*, tool_calls=None, content=None, finish_reason="tool_calls"):
    """Build a minimal mock of openai.types.chat.ChatCompletion."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.tool_calls = tool_calls
    choice.message.content = content
    return choice


def _mock_response(choice):
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNormaliseVerdict(unittest.TestCase):
    def test_false_maps_to_contradicted(self):
        self.assertEqual(_normalise_verdict("FALSE"), "CONTRADICTED")

    def test_true_maps_to_supported(self):
        self.assertEqual(_normalise_verdict("TRUE"), "SUPPORTED")

    def test_case_insensitive(self):
        self.assertEqual(_normalise_verdict("false"), "CONTRADICTED")
        self.assertEqual(_normalise_verdict("contradicted"), "CONTRADICTED")

    def test_passthrough_contradicted(self):
        self.assertEqual(_normalise_verdict("CONTRADICTED"), "CONTRADICTED")

    def test_passthrough_supported(self):
        self.assertEqual(_normalise_verdict("SUPPORTED"), "SUPPORTED")

    def test_passthrough_misleading(self):
        self.assertEqual(_normalise_verdict("MISLEADING"), "MISLEADING")

    def test_passthrough_insufficient(self):
        self.assertEqual(_normalise_verdict("INSUFFICIENT_EVIDENCE"), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(_normalise_verdict("INSUFFICIENT EVIDENCE"), "INSUFFICIENT_EVIDENCE")

    def test_unknown_returns_none(self):
        self.assertIsNone(_normalise_verdict("GARBAGE"))
        self.assertIsNone(_normalise_verdict(""))


class TestVerifiableClaimFilter(unittest.TestCase):
    def test_removes_common_conversational_sentences(self):
        claims = [
            {"id": "thanks", "text": "Thank you very much."},
            {"id": "food", "text": "I want to eat some food."},
            {"id": "fact", "text": "Paris is the capital of France."},
        ]
        filtered = filter_verifiable_claims(claims)
        self.assertEqual(filtered, [{"id": "fact", "text": "Paris is the capital of France."}])


class TestCheckContext(unittest.TestCase):
    def test_normalises_and_limits_context(self):
        context = _normalise_check_context({
            "reference_date": "2020-11-03",
            "location": "United States",
            "speaker": "Campaign spokesperson",
            "notes": "Election-night statement.",
        })
        self.assertEqual(context["reference_date"], "2020-11-03")
        self.assertIn("Evaluate as of: 2020-11-03", _context_summary(context))
        self.assertIn("Location: United States", _context_summary(context))

    def test_rejects_invalid_reference_date(self):
        with self.assertRaises(ValueError):
            _normalise_check_context({"reference_date": "03-11-2020"})

    def test_detects_conflicting_first_person_name(self):
        conflict = _speaker_identity_conflict(
            "My name is Boris and I sell cars.", {"speaker": "Harshad"}
        )
        self.assertEqual(conflict, ("Boris", "Harshad"))

    def test_does_not_flag_matching_first_person_name(self):
        self.assertIsNone(_speaker_identity_conflict(
            "My name is Harshad.", {"speaker": "Harshad"}
        ))


class TestParseXmlToolContent(unittest.TestCase):
    def test_full_xml_parsed(self):
        result = _parse_xml_tool_content(XML_TRUNCATED_CONTENT)
        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertEqual(result["one_liner"], "Earth is an oblate spheroid, not flat.")
        self.assertAlmostEqual(result["confidence"], 0.99)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["url"], "https://nasa.gov")

    def test_truncated_sources_json_salvaged(self):
        result = _parse_xml_tool_content(XML_SEVERELY_TRUNCATED)
        self.assertEqual(result["verdict"], "CONTRADICTED")
        # Sources are truncated mid-JSON; the salvage attempt may return 1 item or []
        # but must NOT raise
        self.assertIsInstance(result["sources"], list)

    def test_false_verdict_normalised(self):
        xml = (
            "<parameter=verdict>\nFALSE\n</parameter>\n"
            "<parameter=one_liner>\nFalse claim.\n</parameter>\n"
        )
        result = _parse_xml_tool_content(xml)
        self.assertEqual(result["verdict"], "CONTRADICTED")

    def test_missing_field_returns_none(self):
        result = _parse_xml_tool_content("<parameter=verdict>\nCONTRADICTED\n</parameter>")
        self.assertIsNone(result["explanation"])
        self.assertIsNone(result["confidence"])


class TestExtractVerdictFromResearch(unittest.TestCase):
    def test_exact_failing_report_returns_contradicted(self):
        """Regression: the Earth-is-flat report must yield CONTRADICTED, not None."""
        verdict = _extract_verdict_from_research(EARTH_FLAT_REPORT)
        self.assertEqual(verdict, "CONTRADICTED")

    def test_verdict_true_returns_supported(self):
        text = "**Verdict: TRUE** — Berlin is the capital of Germany."
        self.assertEqual(_extract_verdict_from_research(text), "SUPPORTED")

    def test_verdict_supported_passthrough(self):
        text = "# Verdict: SUPPORTED\nThe claim is verified."
        self.assertEqual(_extract_verdict_from_research(text), "SUPPORTED")

    def test_verdict_contradicted_passthrough(self):
        text = "Verdict: CONTRADICTED — evidence shows otherwise."
        self.assertEqual(_extract_verdict_from_research(text), "CONTRADICTED")

    def test_no_verdict_returns_none(self):
        self.assertIsNone(_extract_verdict_from_research("Nothing here."))

    def test_fact_check_result_label(self):
        text = '# Fact-Check Result: "The Earth is flat."\n\n**Verdict: FALSE**'
        self.assertEqual(_extract_verdict_from_research(text), "CONTRADICTED")


class TestSourcesFromResearch(unittest.TestCase):
    def test_extracts_markdown_sources(self):
        sources = _sources_from_research(EARTH_FLAT_REPORT)
        urls = [s["url"] for s in sources]
        self.assertIn("https://www.nasa.gov/history/90-years-of-our-changing-views-of-earth", urls)
        self.assertIn("https://en.wikipedia.org/wiki/Spherical_Earth", urls)

    def test_no_fabrication_on_empty_text(self):
        self.assertEqual(_sources_from_research("No URLs here."), [])

    def test_deduplication(self):
        text = "- A: https://example.com\n- B: https://example.com"
        sources = _sources_from_research(text)
        urls = [s["url"] for s in sources]
        self.assertEqual(urls.count("https://example.com"), 1)

    def test_caps_at_ten(self):
        lines = "\n".join(f"- Source {i}: https://example.com/{i}" for i in range(20))
        self.assertLessEqual(len(_sources_from_research(lines)), 10)


class TestStructuredVerdictNemotron(unittest.TestCase):
    """Integration tests using a mocked structured-output client."""

    def _patch_client(self, choice):
        return patch.object(_app, "llm_client", create=True,
                            **{"chat.completions.create.return_value": _mock_response(choice)})

    def test_path1_normal_tool_call(self):
        """Path 1: clean tool_calls JSON → parsed and verdict normalised."""
        tc = MagicMock()
        tc.function.arguments = json.dumps({
            "verdict": "FALSE",        # MiniMax uses FALSE
            "one_liner": "Earth is not flat.",
            "explanation": "Multiple sources confirm this.",
            "confidence": 0.99,
            "sources": [{"title": "NASA", "url": "https://nasa.gov", "supports_claim": False}],
        })
        choice = _mock_choice(tool_calls=[tc], finish_reason="tool_calls")

        with self._patch_client(choice):
            result = structured_verdict_nemotron("The Earth is flat.", EARTH_FLAT_REPORT)

        self.assertEqual(result["verdict"], "CONTRADICTED")   # FALSE → CONTRADICTED
        self.assertAlmostEqual(result["confidence"], 0.99)
        self.assertEqual(len(result["sources"]), 1)

    def test_path2_truncated_xml_content(self):
        """Path 2: finish_reason=length with <parameter=…> XML → XML parser used."""
        choice = _mock_choice(
            tool_calls=None,
            content=XML_TRUNCATED_CONTENT,
            finish_reason="length",
        )

        with self._patch_client(choice):
            result = structured_verdict_nemotron("The Earth is flat.", EARTH_FLAT_REPORT)

        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertIsNotNone(result["explanation"])
        self.assertGreater(result["confidence"], 0)

    def test_path2_truncated_xml_fills_missing_sources_from_research(self):
        """Path 2: when XML sources are empty, fall back to research text URLs."""
        xml_no_sources = (
            "<parameter=verdict>\nCONTRADICTED\n</parameter>\n"
            "<parameter=one_liner>\nFalse.\n</parameter>\n"
            "<parameter=explanation>\nSee sources.\n</parameter>\n"
            "<parameter=confidence>\n0.9\n</parameter>\n"
            "<parameter=sources>\n[]\n</parameter>\n"
        )
        choice = _mock_choice(tool_calls=None, content=xml_no_sources, finish_reason="length")

        with self._patch_client(choice):
            result = structured_verdict_nemotron("The Earth is flat.", EARTH_FLAT_REPORT)

        # Sources should be extracted from the research text
        self.assertGreater(len(result["sources"]), 0)
        urls = [s["url"] for s in result["sources"]]
        self.assertIn("https://www.nasa.gov/history/90-years-of-our-changing-views-of-earth", urls)

    def test_path3_no_tool_call_reads_research_text(self):
        """Path 3: no tool call, no XML → verdict extracted from research report."""
        choice = _mock_choice(
            tool_calls=None,
            content="I cannot determine the verdict.",
            finish_reason="stop",
        )

        with self._patch_client(choice):
            result = structured_verdict_nemotron("The Earth is flat.", EARTH_FLAT_REPORT)

        # EARTH_FLAT_REPORT contains '**Verdict: FALSE**' → CONTRADICTED
        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertGreater(len(result["sources"]), 0)

    def test_path3_truly_unknown_returns_insufficient_evidence(self):
        """Path 3 with no recognisable verdict → INSUFFICIENT_EVIDENCE, never fabricated."""
        choice = _mock_choice(tool_calls=None, content="", finish_reason="stop")

        with self._patch_client(choice):
            result = structured_verdict_nemotron(
                "Some claim.",
                "The research found nothing conclusive.",  # no verdict keyword
            )

        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["sources"], [])   # nothing to fabricate


ARTICLE_URL = "https://www.washingtonpost.com/politics/2024/03/01/senate-vote-story/"
HOMEPAGE_URL = "https://www.washingtonpost.com/"
NASA_URL = "https://www.nasa.gov/history/90-years-of-our-changing-views-of-earth"


def _tool_message(name: str, content):
    from langchain_core.messages import ToolMessage
    return ToolMessage(content=content, tool_call_id="call-1", name=name)


class TestTavilySourceExtraction(unittest.TestCase):
    def test_json_string_preserves_article_url_and_title(self):
        payload = json.dumps({
            "results": [{
                "title": "Senate vote stalls",
                "url": ARTICLE_URL,
                "content": "The Senate voted…",
            }]
        })
        sources = _sources_from_tavily_messages([
            _tool_message("tavily_search", payload),
        ])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], ARTICLE_URL)
        self.assertEqual(sources[0]["title"], "Senate vote stalls")
        self.assertEqual(sources[0]["domain"], "washingtonpost.com")
        self.assertFalse(sources[0]["is_homepage"] if "is_homepage" in sources[0] else _app._is_site_root(sources[0]["url"]))

    def test_dict_content_and_ignores_other_tools(self):
        messages = [
            _tool_message("write_todos", '{"results":[{"title":"nope","url":"https://example.com/x"}]}'),
            _tool_message("tavily_search", {
                "results": [{"title": "NASA history", "url": NASA_URL}],
            }),
        ]
        sources = _sources_from_tavily_messages(messages)
        self.assertEqual([s["url"] for s in sources], [NASA_URL])

    def test_content_blocks(self):
        payload = json.dumps({"results": [{"title": "Wiki", "url": "https://en.wikipedia.org/wiki/Earth"}]})
        sources = _sources_from_tavily_messages([
            _tool_message("tavily_search", [{"type": "text", "text": payload}]),
        ])
        self.assertEqual(sources[0]["url"], "https://en.wikipedia.org/wiki/Earth")


class TestMergeSources(unittest.TestCase):
    def test_drops_homepage_when_article_exists(self):
        tavily = [{
            "title": "Senate vote stalls",
            "url": ARTICLE_URL,
            "from_tavily": True,
            "supports_claim": None,
        }]
        nemotron = [{
            "title": "How the Senate vote reshaped the bill",
            "url": HOMEPAGE_URL,
            "supports_claim": True,
        }]
        report = f"The Washington Post: {HOMEPAGE_URL}\nAlso see {ARTICLE_URL}"
        merged = _merge_sources(nemotron, tavily, report)
        urls = [s["url"] for s in merged]
        self.assertEqual(urls, [ARTICLE_URL])
        self.assertEqual(merged[0]["title"], "Senate vote stalls")
        self.assertFalse(merged[0]["is_homepage"])

    def test_homepage_only_is_labeled_not_an_article(self):
        nemotron = [{
            "title": "How the Senate vote reshaped the bill",
            "url": HOMEPAGE_URL,
            "supports_claim": False,
        }]
        merged = _merge_sources(nemotron, [], "")
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["is_homepage"])
        self.assertEqual(merged[0]["url"], HOMEPAGE_URL)
        self.assertEqual(merged[0]["title"], "washingtonpost.com")

    def test_does_not_keep_url_absent_from_tavily_and_report(self):
        tavily = [{"title": "NASA", "url": NASA_URL, "from_tavily": True}]
        nemotron = [{"title": "Invented", "url": "https://example.com/invented-article", "supports_claim": True}]
        merged = _merge_sources(nemotron, tavily, "See NASA only.")
        self.assertEqual([s["url"] for s in merged], [NASA_URL])

    def test_copies_support_annotation_onto_tavily_url(self):
        tavily = [{"title": "NASA history", "url": NASA_URL, "from_tavily": True}]
        nemotron = [{"title": "NASA", "url": NASA_URL, "supports_claim": True}]
        merged = _merge_sources(nemotron, tavily, "")
        self.assertTrue(merged[0]["supports_claim"])
        self.assertEqual(merged[0]["title"], "NASA history")

    def test_keeps_verbatim_article_from_report_when_tavily_missed_it(self):
        extra = "https://en.wikipedia.org/wiki/Spherical_Earth"
        tavily = [{"title": "NASA history", "url": NASA_URL, "from_tavily": True}]
        nemotron = [{"title": "Spherical Earth", "url": extra, "supports_claim": False}]
        merged = _merge_sources(nemotron, tavily, f"Wikipedia: {extra}")
        urls = [s["url"] for s in merged]
        self.assertIn(NASA_URL, urls)
        self.assertIn(extra, urls)


class TestVerdictCounts(unittest.TestCase):
    def test_counts_each_claim_once_and_ignores_superseded(self):
        claims = [
            {"status": "done", "verdict": "SUPPORTED"},
            {"status": "done", "verdict": "CONTRADICTED"},
            {"status": "done", "verdict": "MISLEADING"},
            {"status": "done", "verdict": "INSUFFICIENT_EVIDENCE"},
            {"status": "researching"},
            {"status": "error", "verdict": "INSUFFICIENT_EVIDENCE"},
            {"status": "superseded", "verdict": "SUPPORTED"},
        ]
        counts = _verdict_counts(claims)
        self.assertEqual(counts["SUPPORTED"], 1)
        self.assertEqual(counts["CONTRADICTED"], 1)
        self.assertEqual(counts["MISLEADING"], 1)
        self.assertEqual(counts["INSUFFICIENT_EVIDENCE"], 1)
        self.assertEqual(counts["PENDING"], 1)
        self.assertEqual(counts["FAILED"], 1)

    def test_done_then_error_transition_is_not_double_counted(self):
        claim = {"status": "done", "verdict": "SUPPORTED"}
        self.assertEqual(_verdict_counts([claim])["SUPPORTED"], 1)
        claim["status"] = "error"
        counts = _verdict_counts([claim])
        self.assertEqual(counts["SUPPORTED"], 0)
        self.assertEqual(counts["FAILED"], 1)


class TestBuildPdf(unittest.TestCase):
    def _snapshot(self):
        long_explanation = ("Evidence paragraph. " * 80).strip()
        claims = []
        verdicts = [
            "SUPPORTED", "CONTRADICTED", "MISLEADING", "INSUFFICIENT_EVIDENCE",
            "SUPPORTED", "CONTRADICTED",
        ]
        for i, verdict in enumerate(verdicts):
            claims.append({
                "text": f"Extracted claim {i} & <details>",
                "sentence": f"The speaker said sentence {i}.",
                "status": "done",
                "verdict": verdict,
                "one_liner": f"One liner {i}",
                "explanation": long_explanation,
                "confidence": 0.8,
                "sources": [{
                    "title": "NASA changing views",
                    "url": NASA_URL + f"?q={i}&x=1",
                    "domain": "nasa.gov",
                    "is_homepage": False,
                    "supports_claim": True,
                }] if i % 2 == 0 else [],
            })
        claims[0]["check_context"] = {
            "reference_date": "2020-11-03",
            "location": "United States",
        }
        claims.append({
            "text": "Still checking this.",
            "sentence": "I think this might be true.",
            "status": "researching",
            "verdict": None,
            "sources": [],
        })
        claims.append({
            "text": "Lookup failed.",
            "sentence": "Something went wrong in the sentence.",
            "status": "error",
            "verdict": "INSUFFICIENT_EVIDENCE",
            "explanation": "timeout & <broken>",
            "sources": [],
        })
        return claims

    def test_pdf_has_sections_links_and_multiple_pages(self):
        pdf = _build_pdf(self._snapshot())
        self.assertTrue(pdf.startswith(b"%PDF"))
        page_count = len(re.findall(rb"/Type\s*/Page(?!s)", pdf))
        self.assertGreaterEqual(page_count, 2)
        self.assertIn(b"/URI", pdf)
        self.assertIn(NASA_URL.encode(), pdf)

        import base64
        import zlib
        texts = []
        for match in re.finditer(rb"stream\r?\n", pdf):
            start = match.end()
            end = pdf.find(b"endstream", start)
            raw = pdf[start:end].strip()
            try:
                if b"~>" in raw:
                    raw = raw[:raw.rfind(b"~>")]
                texts.append(zlib.decompress(base64.a85decode(raw)).decode("latin1"))
            except Exception:
                texts.append(raw.decode("latin1", errors="ignore"))
        blob = "\n".join(texts)
        self.assertIn("Fact-Check Report", blob)
        self.assertIn("Supported", blob)
        self.assertIn("Insufficient evidence", blob)
        self.assertIn("Pending", blob)
        self.assertIn("Failed", blob)
        self.assertIn("Spoken sentence", blob)
        self.assertIn("The speaker said sentence 0.", blob)
        self.assertIn("Extracted claim 0", blob)
        self.assertIn("Fact-check context", blob)
        self.assertIn("Evaluate as of: 2020-11-03", blob)
        self.assertIn("Research in progress", blob)
        self.assertIn("Research failed", blob)
        self.assertIn("No sources were returned", blob)
        self.assertIn("Research failed. This check is not completed. timeout &", blob)
        self.assertIn("(broken)", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
