import tempfile
import unittest
from pathlib import Path

from AI import Agent, LLMProvider, OllamaProvider


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
