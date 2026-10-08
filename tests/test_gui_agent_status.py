import re
import unittest

from agente_gui import AgentGUI


class FakeTreeview:
    def __init__(self):
        self.rows = {}
        self.next_id = 0

    def insert(self, _parent, _index, *, values, tags):
        self.next_id += 1
        item_id = f"item-{self.next_id}"
        self.rows[item_id] = {"values": values, "tags": tags}
        return item_id

    def item(self, item_id, option=None, **kwargs):
        if kwargs:
            self.rows[item_id].update(kwargs)
        if option:
            return self.rows[item_id][option]
        return self.rows[item_id]

    def get_children(self):
        return tuple(self.rows)

    def delete(self, item_id):
        del self.rows[item_id]


class GuiAgentStatusTests(unittest.TestCase):
    def setUp(self):
        self.gui = AgentGUI.__new__(AgentGUI)
        self.gui.agent_list = FakeTreeview()
        self.gui.agent_rows = {}
        self.gui.output_lines = []
        self.gui._aggiungi_output = self.gui.output_lines.append

    def test_output_identifies_timestamp_and_agent_for_search_events(self):
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "tool_result",
                "agent_id": "researcher",
                "title": "Agente di ricerca",
                "tool_name": "web_search",
                "message": "Risultati trovati",
            }
        )

        self.assertEqual(len(self.gui.output_lines), 1)
        self.assertRegex(
            self.gui.output_lines[0],
            re.compile(
                r"^🔎 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} · "
                r"Agente di ricerca · web_search\n"
                r"   Risultato strumento: Risultati trovati\n\n$"
            ),
        )

    def test_output_identifies_orchestrator_for_mode_event(self):
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "mode_selected",
                "message": "Esecuzione diretta",
            }
        )

        self.assertRegex(
            self.gui.output_lines[0],
            r"^⚙️ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} · "
            r"Orchestratore\n   Esecuzione diretta\n\n$",
        )

    def test_multiline_tool_output_is_indented_in_a_separate_block(self):
        self.gui._aggiungi_output_etichettato(
            "Agente di ricerca",
            "Risultati strumento:\n- prima fonte\n- seconda fonte",
        )

        self.assertRegex(
            self.gui.output_lines[0],
            r"^🔎 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} · "
            r"Agente di ricerca\n"
            r"   Risultati strumento:\n"
            r"   - prima fonte\n"
            r"   - seconda fonte\n\n$",
        )

    def test_agent_status_updates_existing_row_and_color_tag(self):
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "agent_started",
                "agent_id": "planner",
                "title": "Pianificatore",
            }
        )
        row_id = self.gui.agent_rows["planner"]
        self.assertEqual(self.gui.agent_list.item(row_id, "tags"), ("running",))
        self.assertEqual(self.gui.agent_list.item(row_id, "values")[2], "In esecuzione")

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "agent_completed",
                "agent_id": "planner",
                "title": "Pianificatore",
            }
        )
        self.assertEqual(self.gui.agent_rows["planner"], row_id)
        self.assertEqual(self.gui.agent_list.item(row_id, "tags"), ("completed",))
        self.assertEqual(self.gui.agent_list.item(row_id, "values")[2], "Completato")

    def test_failed_task_is_shown_as_red_error_row(self):
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "task_failed",
                "task_id": "lookup",
                "title": "Ricerca offerte",
                "error": "HTTP 403",
            }
        )

        row_id = self.gui.agent_rows["task:lookup"]
        self.assertEqual(self.gui.agent_list.item(row_id, "tags"), ("failed",))
        self.assertEqual(self.gui.agent_list.item(row_id, "values")[2], "Errore: HTTP 403")

    def test_clears_previous_request_rows(self):
        self.gui._imposta_stato_agente("planner", "Pianificatore", "completed")

        self.gui._pulisci_lista_agenti()

        self.assertEqual(self.gui.agent_rows, {})
        self.assertEqual(self.gui.agent_list.get_children(), ())


if __name__ == "__main__":
    unittest.main()
