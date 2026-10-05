import os
import tempfile
import unittest
from pathlib import Path

from AI import Agent, LLMProvider, LLMResponse, OllamaProvider, ToolCall


class DummyProvider(LLMProvider):
    def complete(self, messages, tools=None):
        raise NotImplementedError


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp_dir.name)
        self.agent = Agent(DummyProvider(), sandbox=self.sandbox)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_registers_nine_sandbox_tools(self):
        self.assertEqual(
            set(self.agent.tools),
            {
                "list_files",
                "search_text",
                "read_file_excerpt",
                "create_file",
                "append_to_file",
                "move_file",
                "extract_information",
                "compare_files",
                "prepare_tasks",
            },
        )

    def test_registers_script_tool_only_with_explicit_approval_handler(self):
        enabled_agent = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            script_approval=lambda _path, _script: False,
        )

        self.assertIn("run_bash_script", enabled_agent.tools)

    def test_tool_allowlist_preserves_only_allowed_tools(self):
        allowed_agent = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            script_approval=lambda _path, _script: False,
            allowed_tools={"list_files", "read_webpage"},
        )
        allowed_agent.add_tool(lambda: "web", name="read_webpage")
        allowed_agent.add_tool(lambda: "unsafe", name="write_webpage")

        self.assertEqual(set(allowed_agent.tools), {"list_files", "read_webpage"})

    def test_tool_restrictions_can_only_be_narrowed(self):
        self.agent.restrict_tools({"list_files", "create_file"})
        self.agent.restrict_tools({"create_file", "read_file_excerpt"})
        self.agent.add_tool(lambda: None, name="read_file_excerpt")

        self.assertEqual(set(self.agent.tools), {"create_file"})

    def test_lists_searches_and_reads_files(self):
        (self.sandbox / "note.txt").write_text(
            "Prima riga\nDa fare: inviare il report\nUltima riga", encoding="utf-8"
        )

        self.assertEqual(self.agent.list_files(), ["note.txt"])
        self.assertEqual(
            self.agent.search_text("REPORT"),
            ["note.txt:2: Da fare: inviare il report"],
        )
        self.assertEqual(
            self.agent.read_file_excerpt("note.txt", 2, 2),
            "2: Da fare: inviare il report",
        )

    def test_creates_appends_and_moves_files(self):
        self.agent.create_file("reports/summary.txt", "Risultato")
        self.agent.append_to_file("reports/summary.txt", " finale")
        self.agent.move_file("reports/summary.txt", "archive/summary.txt")

        self.assertEqual(
            (self.sandbox / "archive/summary.txt").read_text(encoding="utf-8"),
            "Risultato finale",
        )
        self.assertFalse((self.sandbox / "reports/summary.txt").exists())

    def test_file_name_collision_is_reported_without_overwriting(self):
        class DuplicateFileProvider(LLMProvider):
            def __init__(self):
                self.responses = iter(
                    [
                        LLMResponse(
                            tool_calls=[
                                ToolCall(
                                    name="create_file",
                                    arguments={
                                        "file_path": "duplicate.txt",
                                        "content": "prima versione",
                                    },
                                ),
                                ToolCall(
                                    name="create_file",
                                    arguments={
                                        "file_path": "duplicate.txt",
                                        "content": "versione duplicata",
                                    },
                                ),
                            ]
                        ),
                        LLMResponse(content="Il secondo file non è stato creato."),
                    ]
                )

            def complete(self, messages, tools=None):
                return next(self.responses)

        results = []
        agent = Agent(DuplicateFileProvider(), sandbox=self.sandbox)
        agent.send("Crea due volte lo stesso file")

        result = agent.run(on_tool_result=lambda name, value: results.append((name, value)))

        self.assertEqual(result, "Il secondo file non è stato creato.")
        self.assertEqual(
            (self.sandbox / "duplicate.txt").read_text(encoding="utf-8"),
            "prima versione",
        )
        self.assertIn("non è stato sovrascritto", results[1][1])
        self.assertEqual(len(results), 2)

    def test_extracts_information_compares_files_and_prepares_tasks(self):
        (self.sandbox / "first.txt").write_text(
            "Owner: Ada\n- [ ] Inviare il report\nNota generale", encoding="utf-8"
        )
        (self.sandbox / "second.txt").write_text(
            "Owner: Ada\n- [x] Inviare il report\nNota aggiornata", encoding="utf-8"
        )

        self.assertEqual(
            self.agent.extract_information("first.txt", "owner"),
            [{"line": 1, "text": "Owner: Ada"}],
        )
        self.assertIn("-Nota generale", self.agent.compare_files("first.txt", "second.txt"))
        self.assertEqual(self.agent.prepare_tasks("first.txt"), ["Inviare il report"])

    def test_rejects_paths_outside_sandbox(self):
        with self.assertRaises(PermissionError):
            self.agent.read_file_excerpt("../outside.txt")

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "Script execution as root is disabled")
    def test_runs_approved_script_and_streams_output(self):
        approved_script = "printf 'esecuzione approvata\\n'\n"
        approvals = []
        output = []
        temporary_paths = []

        def approve(path, script):
            self.assertEqual(script, approved_script)
            self.assertEqual(path, "script temporaneo (script.sh)")
            candidates = [
                directory / "script.sh"
                for directory in Path(tempfile.gettempdir()).glob("agente-script-*")
                if (directory / "script.sh").read_text(encoding="utf-8") == approved_script
            ]
            self.assertTrue(candidates)
            temporary_script = candidates[0]
            self.assertEqual(temporary_script.read_text(encoding="utf-8"), approved_script)
            temporary_paths.append(temporary_script)
            approvals.append((path, script))
            return True

        agent = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            script_approval=approve,
            script_output=output.append,
        )

        result = agent.run_bash_script(approved_script)

        self.assertEqual(approvals, [("script temporaneo (script.sh)", approved_script)])
        self.assertEqual("".join(output), "esecuzione approvata\n")
        self.assertIn("codice 0", result)
        self.assertFalse(temporary_paths[0].exists())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "Script execution as root is disabled")
    def test_executes_approved_snapshot_and_honors_rejection(self):
        approved_script = "printf 'snapshot\\n'\n"
        output = []
        approved = []
        temporary_paths = []

        def approve_script(path, script):
            self.assertEqual(path, "script temporaneo (script.sh)")
            self.assertEqual(script, approved_script)
            temporary_paths.extend(
                directory / "script.sh"
                for directory in Path(tempfile.gettempdir()).glob("agente-script-*")
                if (directory / "script.sh").read_text(encoding="utf-8") == script
            )
            approved.append(True)
            return True

        agent = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            script_approval=approve_script,
            script_output=output.append,
        )

        agent.run_bash_script(approved_script)
        self.assertEqual("".join(output), "snapshot\n")
        self.assertTrue(approved)
        self.assertTrue(temporary_paths)
        self.assertFalse(temporary_paths[-1].exists())

        denied_temporaries = []

        def reject_script(_path, _script):
            denied_temporaries.extend(
                directory / "script.sh"
                for directory in Path(tempfile.gettempdir()).glob("agente-script-*")
                if (directory / "script.sh").read_text(encoding="utf-8") == approved_script
            )
            return False

        denied_agent = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            script_approval=reject_script,
        )
        self.assertIn("annullata", denied_agent.run_bash_script(approved_script))
        self.assertTrue(denied_temporaries)
        self.assertFalse(denied_temporaries[-1].exists())

    def test_restricts_script_content(self):
        agent = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            script_approval=lambda _path, _script: True,
        )

        with self.assertRaises(ValueError):
            agent.run_bash_script("x" * (64 * 1024 + 1))
        with self.assertRaises(ValueError):
            agent.run_bash_script("echo '\x00'")
        with self.assertRaises(TypeError):
            agent.run_bash_script(42)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "Script execution as root is disabled")
    def test_stops_scripts_that_exceed_the_output_limit(self):
        temporary_scripts = []

        def approve(_path, script):
            temporary_scripts.extend(
                directory / "script.sh"
                for directory in Path(tempfile.gettempdir()).glob("agente-script-*")
                if (directory / "script.sh").read_text(encoding="utf-8") == script
            )
            return True

        agent = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            script_approval=approve,
        )

        with self.assertRaisesRegex(RuntimeError, "limite di output"):
            agent.run_bash_script("printf '%1100000s' ''\n")
        self.assertTrue(temporary_scripts)
        self.assertFalse(temporary_scripts[-1].exists())

    def test_custom_action_replaces_default(self):
        customized = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            actions={
                "read_file_excerpt": lambda file_path, start_line=1, end_line=50: "personalizzato"
            },
        )

        self.assertEqual(
            customized.tools["read_file_excerpt"](file_path="note.txt"),
            "personalizzato",
        )

    def test_tool_schema_preserves_input_types_and_custom_names(self):
        customized = Agent(
            DummyProvider(),
            sandbox=self.sandbox,
            actions={
                "read_file_excerpt": lambda file_path, start_line=1, end_line=50: "personalizzato"
            },
        )
        schema = OllamaProvider("test")._tool_to_schema(
            customized.tools["read_file_excerpt"]
        )
        self.assertEqual(schema["function"]["name"], "read_file_excerpt")
        self.assertEqual(
            schema["function"]["parameters"]["properties"]["start_line"]["type"],
            "number",
        )
        self.assertEqual(schema["function"]["parameters"]["required"], ["file_path"])

    def test_rejects_unknown_custom_action(self):
        with self.assertRaisesRegex(ValueError, "non riconosciute"):
            Agent(DummyProvider(), sandbox=self.sandbox, actions={"not_a_tool": lambda: None})


if __name__ == "__main__":
    unittest.main()
