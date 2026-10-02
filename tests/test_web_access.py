import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import agente as agente_module

from AI import Agent, InternetAccess, LLMProvider, LLMResponse
from AI.core import ToolCall
from AI.web import (
    MAX_PAGE_BYTES,
    MAX_REDIRECTS,
    _SearchResultsParser,
    _VisibleTextParser,
    _unwrap_search_url,
)


class DummyProvider(LLMProvider):
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, messages, tools=None):
        return next(self.responses)


class WebAccessTests(TestCase):
    def setUp(self):
        self.internet = InternetAccess()

    def test_search_parser_extracts_https_result_and_sanitized_snippet(self):
        parser = _SearchResultsParser()
        parser.feed(
            """
            <div class="result">
              <div class="result__title"><a class="result__a"
                href="https://example.org/article">Useful result</a></div>
              <div class="result__snippet">Public <b>text</b> snippet</div>
              <script>ignore this code</script>
            </div>
            """
        )

        self.assertEqual(len(parser.results), 1)
        self.assertEqual(parser.results[0].title, "Useful result")
        self.assertEqual(parser.results[0].snippet, "Public text snippet")
        self.assertEqual(parser.results[0].url, "https://example.org/article")

    def test_search_parser_extracts_duckduckgo_lite_results(self):
        parser = _SearchResultsParser()
        parser.feed(
            """
            <a rel="nofollow" class="result-link"
              href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Farticle">
              Useful Lite result
            </a>
            <td class="result-snippet">Public <b>text</b> snippet</td>
            """
        )

        self.assertEqual(len(parser.results), 1)
        self.assertEqual(parser.results[0].title, "Useful Lite result")
        self.assertEqual(parser.results[0].snippet, "Public text snippet")
        self.assertEqual(parser.results[0].url, "https://example.org/article")

    def test_page_parser_excludes_scripts_styles_and_markup(self):
        parser = _VisibleTextParser()
        parser.feed(
            "<h1>Useful page</h1><script>malicious code</script>"
            "<p>Visible <b>text</b></p><style>hidden style</style>"
        )

        self.assertEqual("".join(parser.parts).strip(), "Useful page\nVisible text")

    def test_read_webpage_returns_only_sanitized_visible_text_and_records_source(self):
        body = (
            "<h1>Public information</h1><script>ignore secret code</script>"
            "<p>Contact ada@example.org</p>"
        )
        with patch.object(
            self.internet,
            "_fetch_text",
            return_value=(body, "text/html", "https://example.org/page"),
        ):
            result = self.internet.read_webpage("https://example.org/page")

        self.assertIn("Public information", result)
        self.assertIn("[indirizzo email rimosso]", result)
        self.assertNotIn("ignore secret code", result)
        self.assertEqual(
            self.internet.sources(),
            [("example.org", "https://example.org/page")],
        )

    def test_search_includes_search_results_and_saves_sources(self):
        page = """
        <div class="result">
          <a class="result__a" href="https://example.org/article">Useful result</a>
          <div class="result__snippet">Snippet from the result</div>
        </div>
        """
        with patch.object(
            self.internet,
            "_fetch_text",
            return_value=(
                page,
                "text/html",
                "https://html.duckduckgo.com/html/",
            ),
        ) as fetch:
            result = self.internet.web_search("public topic")

        self.assertIn("https://example.org/article", result)
        self.assertIn("Snippet from the result", result)
        self.assertEqual(
            fetch.call_args.args[0],
            "https://html.duckduckgo.com/html/?q=public+topic",
        )
        self.assertEqual(
            self.internet.sources(),
            [("Useful result", "https://example.org/article")],
        )

    def test_search_falls_back_to_duckduckgo_lite_when_html_has_no_results(self):
        lite_page = """
        <a class="result-link" href="https://example.org/article">Useful Lite result</a>
        <td class="result-snippet">Snippet from Lite</td>
        """
        with patch.object(
            self.internet,
            "_fetch_text",
            side_effect=[
                ("<html><body>challenge</body></html>", "text/html", ""),
                (lite_page, "text/html", ""),
            ],
        ) as fetch:
            result = self.internet.web_search("public topic")

        self.assertIn("Useful Lite result", result)
        self.assertIn("Snippet from Lite", result)
        self.assertIn(
            "https://lite.duckduckgo.com/lite/?q=public+topic",
            [call.args[0] for call in fetch.call_args_list],
        )
        self.assertEqual(
            self.internet.sources(),
            [("Useful Lite result", "https://example.org/article")],
        )

    def test_search_refuses_personal_or_secret_queries(self):
        for query in (
            "contact ada@example.org",
            "api_key=secret-value",
            "ghp_123456789012345678901234567890123456",
        ):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.internet.web_search(query)

    def test_search_refuses_local_machine_paths(self):
        with self.assertRaisesRegex(ValueError, "dati personali"):
            self.internet.web_search("/Users/example/private/report")

    def test_url_validation_rejects_non_https_local_and_nonstandard_ports(self):
        for url in (
            "http://example.org",
            "https://127.0.0.1/private",
            "https://localhost/",
            "https://user:pass@example.org/",
            "https://example.org:8443/",
        ):
            with self.subTest(url=url), self.assertRaises((ValueError, PermissionError)):
                self.internet._validate_public_url(url)

    def test_url_validation_rejects_dns_pointing_to_private_address(self):
        records = [
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ]
        with patch("AI.web.socket.getaddrinfo", return_value=records):
            with self.assertRaises(PermissionError):
                self.internet._validate_public_url("https://example.org/")

    def test_url_validation_rejects_sensitive_query_parameters(self):
        with self.assertRaisesRegex(ValueError, "credenziali"):
            self.internet._validate_public_url(
                "https://example.org/?access_token=secret"
            )

    def test_request_uses_only_get_and_rejects_file_before_reading_body(self):
        connection = MagicMock()
        response = MagicMock()
        response.status = 200
        response.getheaders.return_value = [
            ("Content-Type", "application/octet-stream"),
            ("Content-Disposition", "attachment; filename=program.py"),
        ]
        response.getheader.return_value = "10"
        connection.getresponse.return_value = response

        with (
            patch.object(
                self.internet,
                "_validate_public_url",
                return_value=(
                    urlsplit("https://example.org/program.py"),
                    "example.org",
                    "93.184.216.34",
                ),
            ),
            patch("AI.web._PinnedHTTPSConnection", return_value=connection),
        ):
            with self.assertRaisesRegex(ValueError, "Tipo di contenuto non consentito"):
                self.internet._request_once("https://example.org/program.py")

        connection.request.assert_called_once()
        self.assertEqual(connection.request.call_args.args[:2], ("GET", "/program.py"))
        self.assertNotIn(
            "Authorization",
            connection.request.call_args.kwargs["headers"],
        )
        response.read.assert_not_called()

    def test_request_rejects_oversized_html_before_reading_body(self):
        connection = MagicMock()
        response = MagicMock()
        response.status = 200
        response.getheaders.return_value = [("Content-Type", "text/html")]
        response.getheader.return_value = str(MAX_PAGE_BYTES + 1)
        connection.getresponse.return_value = response

        with (
            patch.object(
                self.internet,
                "_validate_public_url",
                return_value=(
                    urlsplit("https://example.org/"),
                    "example.org",
                    "93.184.216.34",
                ),
            ),
            patch("AI.web._PinnedHTTPSConnection", return_value=connection),
        ):
            with self.assertRaisesRegex(ValueError, "supera il limite"):
                self.internet._request_once("https://example.org/")

        response.read.assert_not_called()

    def test_redirect_count_is_bounded(self):
        with patch.object(
            self.internet,
            "_request_once",
            return_value=(
                302,
                {"location": "https://example.org/next"},
                b"",
            ),
        ) as request:
            with self.assertRaisesRegex(RuntimeError, "reindirizzamenti"):
                self.internet._fetch_text("https://example.org/")

        self.assertEqual(request.call_count, MAX_REDIRECTS + 1)

    def test_search_redirect_links_only_unwrap_to_https(self):
        self.assertEqual(
            _unwrap_search_url(
                "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage"
            ),
            "https://example.org/page",
        )
        self.assertEqual(
            _unwrap_search_url("https://duckduckgo.com/l/?uddg=http%3A%2F%2Fevil.test"),
            "",
        )

    def test_agent_appends_sources_to_final_answer(self):
        provider = DummyProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(name="web_search", arguments={"query": "public topic"})
                    ]
                ),
                LLMResponse(content="Ecco la sintesi."),
            ]
        )
        agent = Agent(provider)
        agent.add_tool(lambda query: "risultati", name="web_search")
        agent.add_source_provider(
            lambda: [("Fonte ufficiale", "https://example.org/source")]
        )
        agent.send("Cerca informazioni.")

        result = agent.run()

        self.assertIn("Ecco la sintesi.", result)
        self.assertIn("Fonti web:", result)
        self.assertIn("https://example.org/source", result)
        self.assertEqual(agent.messages[-1]["role"], "assistant")
        self.assertEqual(agent.messages[-1]["content"], result)

    def test_application_agent_registers_web_tools_with_sandbox_tools(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            agente_module,
            "LMStudioProvider",
            return_value=DummyProvider([]),
        ):
            agent = agente_module.crea_agente(Path(directory))

        self.assertIn("web_search", agent.tools)
        self.assertIn("read_webpage", agent.tools)
        self.assertIn("create_file", agent.tools)
        self.assertNotIn("run_bash_script", agent.tools)
        self.assertEqual(len(agent.tools), 11)
        self.assertEqual(len(agent.source_providers), 1)

    def test_application_agent_exposes_script_runner_only_with_gui_approval(self):
        approval = lambda _path, _script: False
        provider = DummyProvider([])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            agente_module,
            "LMStudioProvider",
            return_value=provider,
        ):
            agent = agente_module.crea_agente(
                Path(directory),
                script_approval=approval,
            )

        self.assertIn("run_bash_script", agent.tools)
        description = agent.tools["run_bash_script"].__doc__
        self.assertIn("nmap", description)
        self.assertIn("file temporaneo", description)
        self.assertIn("direttamente il codice", description)
        self.assertIn("127.0.0.1", agent.messages[0]["content"])

    def test_cli_prompt_does_not_claim_script_execution_is_available(self):
        provider = DummyProvider([])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            agente_module,
            "LMStudioProvider",
            return_value=provider,
        ):
            agent = agente_module.crea_agente(Path(directory))

        self.assertNotIn("run_bash_script", agent.tools)
        self.assertIn("non è disponibile uno strumento di esecuzione script", agent.messages[0]["content"])

    def test_tool_schema_exposes_no_write_or_http_mutation_tool(self):
        self.assertEqual(
            set(self.internet.__class__.__dict__) & {"web_search", "read_webpage"},
            {"web_search", "read_webpage"},
        )
        self.assertFalse(
            any(
                name in {"POST", "PUT", "PATCH", "DELETE", "upload", "download"}
                for name in dir(self.internet)
            )
        )


if __name__ == "__main__":
    unittest.main()
