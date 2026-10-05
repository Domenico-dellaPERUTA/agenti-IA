import unittest

from agente_gui import AgenteGUI


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
        self.gui = AgenteGUI.__new__(AgenteGUI)
        self.gui.agent_list = FakeTreeview()
        self.gui.agent_rows = {}
        self.gui._aggiungi_output = lambda _text: None

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
