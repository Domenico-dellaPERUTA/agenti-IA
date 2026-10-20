import json
import queue
import re
import unittest

from agente_gui import AgentGUI


class FakeTreeview:
    def __init__(self):
        self.rows = {}
        self.next_id = 0

    def insert(self, parent, _index, *, values, tags):
        if parent and parent not in self.rows:
            raise ValueError(f"Parent Treeview inesistente: {parent}")
        self.next_id += 1
        item_id = f"item-{self.next_id}"
        self.rows[item_id] = {"parent": parent, "values": values, "tags": tags}
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


class FakeWindow:
    def __init__(self):
        self.callbacks = []

    def after(self, _delay, callback):
        self.callbacks.append(callback)


class FakeVariable:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


class FakeButton:
    def __init__(self):
        self.state = None

    def configure(self, **kwargs):
        self.state = kwargs.get("state")


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

    def test_parallel_task_is_attached_to_existing_planner_node(self):
        root_id = self.gui._nuovo_nodo_esecuzione(
            "execution-root",
            parent_key=None,
            node_type="request",
            label="Richiesta",
        )
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "agent_started",
                "agent_id": "planner",
                "title": "Pianificatore",
            }
        )
        planner_id = self.gui.agent_rows["planner"]

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "task_started",
                "task_id": "lookup",
                "title": "Elenca file",
                "message": "Avviato",
            }
        )

        task_id = self.gui.agent_rows["task:lookup"]
        self.assertEqual(self.gui.agent_list.item(planner_id)["parent"], root_id)
        self.assertEqual(self.gui.agent_list.item(task_id)["parent"], planner_id)

    def test_direct_tool_event_is_attached_to_direct_agent_node(self):
        root_id = self.gui._nuovo_nodo_esecuzione(
            "execution-root",
            parent_key=None,
            node_type="request",
            label="Richiesta",
        )
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "agent_started",
                "agent_id": "direct",
                "title": "Agente diretto",
            }
        )
        direct_id = self.gui.agent_rows["direct"]

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "tool_selected",
                "agent_id": "direct",
                "title": "Agente diretto",
                "message": "list_files({})",
            }
        )

        tool_key = next(key for key in self.gui.agent_rows if key.startswith("tool:"))
        tool_id = self.gui.agent_rows[tool_key]
        self.assertEqual(self.gui.agent_list.item(direct_id)["parent"], root_id)
        self.assertEqual(self.gui.agent_list.item(tool_id)["parent"], direct_id)

    def test_model_response_does_not_mark_agent_complete_before_agent_completed(self):
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "agent_started",
                "agent_id": "direct",
                "title": "Agente diretto",
            }
        )
        agent_id = self.gui.agent_rows["direct"]

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "model_response",
                "agent_id": "direct",
                "title": "Agente diretto",
                "message": "Eseguirò gli strumenti richiesti.",
            }
        )

        self.assertEqual(self.gui.agent_list.item(agent_id, "tags"), ("running",))

    def test_timeline_keeps_each_agent_phase_as_a_separate_event(self):
        events = (
            {
                "type": "agent_started",
                "agent_id": "direct",
                "title": "Agente diretto",
            },
            {
                "type": "agent_status",
                "agent_id": "direct",
                "title": "Agente diretto",
                "status": "llm_request",
                "message": "Invio la richiesta",
            },
            {
                "type": "model_response",
                "agent_id": "direct",
                "title": "Agente diretto",
                "message": "Chiamerò list_files",
            },
            {
                "type": "tool_selected",
                "agent_id": "direct",
                "title": "Agente diretto",
                "message": "list_files({})",
            },
            {
                "type": "tool_result",
                "agent_id": "direct",
                "title": "Agente diretto",
                "tool_name": "list_files",
                "message": "notes.txt",
            },
            {
                "type": "agent_completed",
                "agent_id": "direct",
                "title": "Agente diretto",
            },
        )
        for event in events:
            self.gui._gestisci_evento_orchestratore(event)

        phase_nodes = [
            node
            for key, node in self.gui.execution_nodes.items()
            if key.startswith("event:")
        ]
        self.assertEqual(len(phase_nodes), len(events))
        self.assertEqual(
            [node.label.split(" · ")[1] for node in phase_nodes],
            [event["type"] for event in events],
        )
        self.assertTrue(
            all(node.parent_key == "direct" for node in phase_nodes)
        )
        llm_request = json.loads(phase_nodes[1].input_data.split("\n", 1)[1])
        self.assertEqual(llm_request["status"], "llm_request")
        self.assertEqual(phase_nodes[4].output_data, "notes.txt")

    def test_final_result_from_worker_is_logged_and_added_to_timeline(self):
        self.gui.events = queue.Queue()
        self.gui.window = FakeWindow()
        self.gui.stato = FakeVariable()
        self.gui.pulsante_avvia = FakeButton()
        self.gui.events.put(("risultato", "Risposta finale"))

        self.gui._leggi_eventi()

        self.assertIn("Risposta finale", "".join(self.gui.output_lines))
        result_event = next(
            node
            for key, node in self.gui.execution_nodes.items()
            if key.startswith("event:")
        )
        self.assertIn("orchestration_completed", result_event.label)
        self.assertEqual(result_event.output_data, "Risposta finale")
        self.assertEqual(self.gui.stato.value, "Completato")
        self.assertEqual(self.gui.pulsante_avvia.state, "normal")
        self.assertEqual(len(self.gui.window.callbacks), 1)

    def test_gui_event_error_does_not_discard_later_final_result(self):
        self.gui.events = queue.Queue()
        self.gui.window = FakeWindow()
        self.gui.stato = FakeVariable()
        self.gui.pulsante_avvia = FakeButton()
        self.gui.events.put(("orchestrator_event", "evento malformato"))
        self.gui.events.put(("risultato", "Risposta recuperata"))

        self.gui._leggi_eventi()

        output = "".join(self.gui.output_lines)
        self.assertIn("Errore GUI durante l'elaborazione", output)
        self.assertIn("Risposta recuperata", output)
        self.assertEqual(self.gui.stato.value, "Completato")
        self.assertEqual(self.gui.pulsante_avvia.state, "normal")

    def test_clears_previous_request_rows(self):
        self.gui._imposta_stato_agente("planner", "Pianificatore", "completed")

        self.gui._pulisci_lista_agenti()

        self.assertEqual(self.gui.agent_rows, {})
        self.assertEqual(self.gui.agent_list.get_children(), ())

    def test_pulisci_lista_agenti_discards_queued_events(self):
        self.gui.events = queue.Queue()
        self.gui.events.put(("orchestrator_event", {"type": "agent_started", "agent_id": "stale"}))
        self.gui.events.put(("risultato", "Obsoleto"))

        self.gui._pulisci_lista_agenti()

        self.assertTrue(self.gui.events.empty())

    def test_fatal_error_marks_only_agents_and_tasks_as_failed(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "agent_started", "agent_id": "planner", "title": "Pianificatore"}
        )
        self.gui._gestisci_evento_orchestratore(
            {"type": "task_started", "task_id": "t1", "title": "Task 1"}
        )
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "tool_selected",
                "agent_id": "planner",
                "title": "Pianificatore",
                "message": "list_files({})",
            }
        )

        self.gui._segna_agenti_in_corso_come_errore("interruzione")

        nodes = self.gui.execution_nodes
        # Root, fasi e tool NON devono essere trasformati in "agente" né marcati failed.
        self.assertEqual(nodes["execution-root"].status, "running")
        self.assertEqual(nodes["execution-root"].node_type, "request")
        for key, node in nodes.items():
            if key.startswith("event:") or key.startswith("tool:"):
                self.assertEqual(node.status, "running", f"{key} non dovrebbe fallire")
        # Solo agente e task in running devono essere marcati failed.
        self.assertEqual(nodes["planner"].status, "failed")
        self.assertEqual(nodes["planner"].node_type, "agente")
        self.assertEqual(nodes["task:t1"].status, "failed")
        self.assertEqual(nodes["task:t1"].node_type, "task")

    def test_agent_status_updates_timestamp_without_changing_status(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "agent_started", "agent_id": "direct", "title": "Agente diretto"}
        )
        node = self.gui.execution_nodes["direct"]
        previous_timestamp = node.timestamp

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "agent_status",
                "agent_id": "direct",
                "title": "Agente diretto",
                "status": "llm_request",
                "message": "Invio la richiesta",
            }
        )

        self.assertEqual(node.status, "running")
        self.assertIn("llm_request", node.output_data)
        self.assertGreaterEqual(node.timestamp, previous_timestamp)

    def test_task_status_updates_task_node(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "task_started", "task_id": "abc", "title": "Ricerca"}
        )

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "task_status",
                "agent_id": "task:abc",
                "task_id": "abc",
                "title": "Ricerca",
                "status": "llm_response",
                "message": "Ricevuta risposta",
            }
        )

        node = self.gui.execution_nodes["task:abc"]
        self.assertEqual(node.status, "running")
        self.assertIn("llm_response", node.output_data)

    def test_agent_retry_marks_node_as_retry(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "agent_started", "agent_id": "direct", "title": "Agente diretto"}
        )

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "agent_retry",
                "agent_id": "direct",
                "title": "Agente diretto",
                "message": "Nuovo tentativo richiesto",
            }
        )

        node = self.gui.execution_nodes["direct"]
        self.assertEqual(node.status, "retry")
        item_id = self.gui.agent_rows["direct"]
        self.assertEqual(self.gui.agent_list.item(item_id, "tags"), ("retry",))

    def test_agent_completed_uses_single_timestamp(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "agent_started", "agent_id": "direct", "title": "Agente diretto"}
        )
        started_at = self.gui.execution_nodes["direct"].timestamp

        self.gui._gestisci_evento_orchestratore(
            {"type": "agent_completed", "agent_id": "direct", "title": "Agente diretto"}
        )

        node = self.gui.execution_nodes["direct"]
        # Il timestamp finale è quello dell'evento completed, non una doppia scrittura.
        self.assertEqual(node.status, "completed")
        self.assertGreaterEqual(node.timestamp, started_at)
        item_id = self.gui.agent_rows["direct"]
        self.assertEqual(self.gui.agent_list.item(item_id, "values")[1], "Completato")


if __name__ == "__main__":
    unittest.main()
