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

_normalise_verdict          = _app._normalise_verdict
_parse_xml_tool_content     = _app._parse_xml_tool_content
_extract_verdict_from_research = _app._extract_verdict_from_research
_sources_from_research      = _app._sources_from_research
structured_verdict_nemotron = _app.structured_verdict_nemotron

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
    """Integration tests using mocked Nebius client."""

    def _patch_client(self, choice):
        return patch.object(_app, "nebius_client", create=True,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
