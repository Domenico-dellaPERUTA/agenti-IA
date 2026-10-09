from contextlib import redirect_stdout
from io import StringIO
import os
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import agente as agente_module
from AI import (
    Agent,
    AI,
    InternetAccess,
    LLMProvider,
    LLMResponse,
    ToolExecutionError,
)
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


def make_application(sandbox, provider=None):
    return AI(
        provider or DummyProvider([]),
        sandbox,
        initial_prompt="test prompt",
        base_system_prompt="Sei un assistente utile.",
    )


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

    def test_search_falls_back_to_lite_after_http_error_from_main_layout(self):
        lite_page = """
        <a class="result-link" href="https://example.org/article">Useful Lite result</a>
        <td class="result-snippet">Snippet from Lite</td>
        """
        with patch.object(
            self.internet,
            "_fetch_text",
            side_effect=[
                ToolExecutionError("HTTP 403"),
                (lite_page, "text/html", ""),
            ],
        ) as fetch:
            result = self.internet.web_search("public topic")

        self.assertIn("Useful Lite result", result)
        self.assertEqual(fetch.call_count, 2)

    def test_search_challenge_is_reported_instead_of_generic_empty_results(self):
        challenge = (
            "<html><body>Please email the following code to: "
            "error-lite@example.com Code: c21b</body></html>"
        )
        with patch.object(
            self.internet,
            "_fetch_text",
            return_value=(challenge, "text/html", ""),
        ):
            with self.assertRaisesRegex(
                ToolExecutionError,
                "verifica anti-automazione|limite temporaneo",
            ):
                self.internet.web_search("programmatore Caserta")

    def test_search_challenge_is_reported_instead_of_generic_empty_results(self):
        challenge = (
            "<html><body>Please email the following code to: "
            "error-lite@example.com Code: c21b</body></html>"
        )
        with patch.object(
            self.internet,
            "_fetch_text",
            return_value=(challenge, "text/html", ""),
        ):
            with self.assertRaisesRegex(
                ToolExecutionError,
                "verifica anti-automazione|limite temporaneo",
            ):
                self.internet.web_search("programmatore Caserta")

    def test_unreadable_search_results_are_recoverable_and_agent_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = DummyProvider(
                [
                    LLMResponse(
                        tool_calls=[
                            ToolCall(
                                name="web_search",
                                arguments={"query": "public topic"},
                            ),
                            ToolCall(
                                name="create_file",
                                arguments={
                                    "file_path": "ricerca.txt",
                                    "content": "Nessun annuncio verificato tramite ricerca.\n",
                                },
                            ),
                        ]
                    ),
                    LLMResponse(content="Ho creato il file segnalando il limite della ricerca."),
                ]
            )
            agent = Agent(provider, sandbox=directory)
            agent.add_tool(self.internet.web_search)
            tool_results = []
            agent.send("Cerca sul web e crea un file con i risultati")

            with patch.object(
                self.internet,
                "_parse_search_results",
                return_value=[],
            ):
                answer = agent.run(
                    on_tool_result=lambda name, value: tool_results.append((name, value))
                )

            self.assertIn("non ha restituito risultati leggibili", tool_results[0][1])
            self.assertEqual(tool_results[1][0], "create_file")
            self.assertTrue((Path(directory) / "ricerca.txt").is_file())
            self.assertEqual(
                answer,
                "Ho creato il file segnalando il limite della ricerca.",
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

    def test_http_status_failure_is_a_recoverable_tool_error(self):
        with patch.object(
            self.internet,
            "_request_once",
            return_value=(403, {}, b""),
        ):
            with self.assertRaisesRegex(ToolExecutionError, "HTTP 403"):
                self.internet._fetch_text("https://example.org/")

    def test_http_error_is_reported_to_model_and_remaining_tools_continue(self):
        provider = DummyProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(name="read_webpage", arguments={"url": "https://blocked.example/"}),
                        ToolCall(name="read_webpage", arguments={"url": "https://available.example/"}),
                    ]
                ),
                LLMResponse(content="Ho proseguito con la seconda pagina."),
            ]
        )
        agent = Agent(provider)

        def read_page(url):
            if "blocked" in url:
                raise ToolExecutionError(
                    "La richiesta web è fallita con stato HTTP 403; pagina non letta."
                )
            return "contenuto della pagina"

        agent.add_tool(read_page, name="read_webpage")
        tool_results = []
        agent.send("Leggi le pagine")

        result = agent.run(
            on_tool_result=lambda name, value: tool_results.append((name, value))
        )

        self.assertEqual(result, "Ho proseguito con la seconda pagina.")
        self.assertIn("HTTP 403", tool_results[0][1])
        self.assertEqual(tool_results[1][1], "contenuto della pagina")
        self.assertEqual(
            [message["content"] for message in agent.messages if message["role"] == "tool"],
            [tool_results[0][1], "contenuto della pagina"],
        )

    def test_http_403_does_not_prevent_requested_file_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = DummyProvider(
                [
                    LLMResponse(
                        tool_calls=[
                            ToolCall(
                                name="read_webpage",
                                arguments={"url": "https://blocked.example/"},
                            ),
                            ToolCall(
                                name="create_file",
                                arguments={
                                    "file_path": "annunci.txt",
                                    "content": "Azienda: esempio\nRuolo: tecnico informatico\n",
                                },
                            ),
                        ]
                    ),
                    LLMResponse(content="Ho creato il file usando le informazioni disponibili."),
                ]
            )
            agent = Agent(provider, sandbox=directory)
            internet = InternetAccess()
            agent.add_tool(internet.read_webpage)
            tool_results = []
            agent.send("Leggi la pagina e crea il file richiesto")

            with patch.object(
                internet,
                "_request_once",
                return_value=(403, {}, b""),
            ):
                answer = agent.run(
                    on_tool_result=lambda name, value: tool_results.append((name, value))
                )

            created_file = Path(directory) / "annunci.txt"
            self.assertEqual(
                created_file.read_text(encoding="utf-8"),
                "Azienda: esempio\nRuolo: tecnico informatico\n",
            )
            self.assertIn("HTTP 403", tool_results[0][1])
            self.assertEqual(tool_results[1][0], "create_file")
            self.assertEqual(answer, "Ho creato il file usando le informazioni disponibili.")

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
        with tempfile.TemporaryDirectory() as directory, patch(
            "AI.application.InternetAccess", return_value=self.internet
        ):
            agent = make_application(Path(directory)).create_agent()

        self.assertIn("web_search", agent.tools)
        self.assertIn("read_webpage", agent.tools)
        self.assertIn("create_file", agent.tools)
        self.assertNotIn("run_bash_script", agent.tools)
        self.assertEqual(len(agent.tools), 11)
        self.assertEqual(len(agent.source_providers), 1)

    def test_application_agent_exposes_script_runner_only_with_gui_approval(self):
        approval = lambda _path, _script: False
        provider = DummyProvider([])
        with tempfile.TemporaryDirectory() as directory, patch(
            "AI.application.InternetAccess", return_value=self.internet
        ):
            agent = make_application(Path(directory), provider).create_agent(
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
        with tempfile.TemporaryDirectory() as directory, patch(
            "AI.application.InternetAccess", return_value=self.internet
        ):
            agent = make_application(Path(directory), provider).create_agent()

        self.assertNotIn("run_bash_script", agent.tools)
        self.assertIn("non è disponibile uno strumento di esecuzione script", agent.messages[0]["content"])

    def test_cli_script_approval_requires_explicit_confirmation(self):
        displayed = StringIO()
        with patch("builtins.input", return_value="ESEGUI"), redirect_stdout(displayed):
            approved = agente_module._approva_script_cli(
                "script temporaneo (script.sh)",
                "printf 'scan locale\\n'\n",
            )

        self.assertTrue(approved)
        self.assertIn("printf 'scan locale\\n'", displayed.getvalue())

        with patch("builtins.input", return_value="si"), redirect_stdout(StringIO()):
            self.assertFalse(
                agente_module._approva_script_cli("script.sh", "echo test")
            )

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "Script execution as root is disabled")
    def test_cli_main_enables_script_execution_and_streams_logs(self):
        provider = DummyProvider(
            [
                LLMResponse(content='{"mode":"direct","tasks":[]}'),
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            name="run_bash_script",
                            arguments={"script": "printf 'cli output\\n'\n"},
                        )
                    ]
                ),
                LLMResponse(content="Scansione completata."),
            ]
        )
        displayed = StringIO()
        with (
            patch.object(application_module, "LMStudioProvider", return_value=provider),
            patch("builtins.input", side_effect=["Scansiona localhost", "ESEGUI"]),
            redirect_stdout(displayed),
        ):
            agente_module.main()

        self.assertIn("printf 'cli output\\n'", displayed.getvalue())
        self.assertIn("cli output", displayed.getvalue())
        self.assertIn("Agente diretto · run_bash_script", displayed.getvalue())
        self.assertIn("Scansione completata.", displayed.getvalue())
        self.assertIn("💬 ", displayed.getvalue())
        self.assertIn("⚙️ ", displayed.getvalue())

    def test_cli_main_uses_timestamped_agent_blocks_without_duplicate_final_answer(self):
        provider = DummyProvider(
            [
                LLMResponse(content='{"mode":"direct","tasks":[]}'),
                LLMResponse(content="Risposta di prova."),
            ]
        )
        displayed = StringIO()
        with (
            patch.object(application_module, "LMStudioProvider", return_value=provider),
            patch("builtins.input", side_effect=["Richiesta di prova", ""]),
            redirect_stdout(displayed),
        ):
            agente_module.main()

        output = displayed.getvalue()
        self.assertRegex(
            output,
            r"💬 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} · Utente\n"
            r"   Richiesta di prova",
        )
        self.assertRegex(
            output,
            r"⚙️ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} · Risultato finale\n"
            r"   Risposta di prova\.",
        )
        self.assertIn("Pianificatore", output)
        self.assertIn("Esecuzione diretta", output)
        self.assertEqual(output.count("Risposta di prova."), 1)

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
