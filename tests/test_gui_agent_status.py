import importlib.util
import json
import os
import queue
import re
import unittest
from unittest import mock

from AI import LLMProvider, LMStudioProvider, OpenAIProvider, OllamaProvider
import agente_gui as agente_gui_module
from agente_gui import AgentGUI, _colonna_tk, _spazi_markdown


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

    def test_failed_orchestration_shows_final_error_block_with_role_names(self):
        from AI import OrchestrationResult
        from AI.orchestration import OrchestrationFailure

        self.gui.events = queue.Queue()
        self.gui.window = FakeWindow()
        self.gui.stato = FakeVariable()
        self.gui.pulsante_avvia = FakeButton()
        risultato = OrchestrationResult(
            answer=None,
            tasks=[],
            mode="direct",
            failures=[OrchestrationFailure(role="planner", error="JSON non valido")],
        )
        self.gui.events.put(("risultato", risultato))

        self.gui._leggi_eventi()

        output = "".join(self.gui.output_lines)
        # Il blocco finale si mostra come errore, non come risposta riuscita.
        self.assertIn("⚠️", output)
        self.assertIn("Risposta finale", output)
        self.assertIn("fasi in errore", output)
        # Il ruolo interno 'planner' è mappato sul nome leggibile.
        self.assertIn("Pianificatore", output)
        self.assertEqual(self.gui.stato.value, "Completato con errori")
        self.assertEqual(self.gui.agent_list.item(
            self.gui.agent_rows["execution-root"], "values"
        )[1], "Errore")

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
                self.assertNotEqual(
                    node.status, "failed", f"{key} non dovrebbe fallire"
                )
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

    def test_tool_result_closes_the_matching_open_tool_node(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "agent_started", "agent_id": "direct", "title": "Agente diretto"}
        )
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "tool_selected",
                "agent_id": "direct",
                "title": "Agente diretto",
                "message": "read_file_excerpt({'path': 'note.txt'})",
            }
        )
        tool_key = next(key for key in self.gui.agent_rows if key.startswith("tool:"))

        self.gui._gestisci_evento_orchestratore(
            {
                "type": "tool_result",
                "agent_id": "direct",
                "title": "Agente diretto",
                "tool_name": "read_file_excerpt",
                "message": "contenuto del file",
            }
        )

        # Il risultato aggiorna il nodo aperto: niente riga tool doppia e lo
        # stato passa a completato, così non resta "In esecuzione" in eterno.
        tool_keys = [key for key in self.gui.agent_rows if key.startswith("tool:")]
        self.assertEqual(tool_keys, [tool_key])
        node = self.gui.execution_nodes[tool_key]
        self.assertEqual(node.status, "completed")
        self.assertIn("read_file_excerpt", node.input_data)
        self.assertEqual(node.output_data, "contenuto del file")
        self.assertEqual(node.activity, "")

    def test_instantaneous_phases_are_registered_not_running(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "mode_selected", "message": "Esecuzione diretta"}
        )

        node = next(
            node
            for key, node in self.gui.execution_nodes.items()
            if key.startswith("event:")
        )
        self.assertEqual(node.status, "registered")
        self.assertEqual(
            self.gui._formatta_stato_compatibile(node.status), "Registrato"
        )

    def test_new_request_clears_pending_tool_nodes(self):
        self.gui._pulisci_lista_agenti()
        self.gui._gestisci_evento_orchestratore(
            {"type": "agent_started", "agent_id": "direct", "title": "Agente diretto"}
        )
        self.gui._gestisci_evento_orchestratore(
            {
                "type": "tool",
                "agent_id": "direct",
                "title": "Agente diretto",
                "message": "list_files({})",
            }
        )

        self.gui._pulisci_lista_agenti()

        self.assertEqual(self.gui._strumenti_in_corso, {})


class GuiMarkdownTests(unittest.TestCase):
    @staticmethod
    def _tag_per_linea(spazi, indice=0):
        return [(c1, c2, tag) for riga, c1, c2, tag in spazi if riga == indice]

    def test_riga_senza_markup_non_produce_spazi(self):
        self.assertEqual(_spazi_markdown(["nessun formato qui"]), [])

    def test_intestazione_livello_due_elide_i_cancelletti(self):
        spazi = self._tag_per_linea(_spazi_markdown(["## Titolo"]))
        self.assertIn((0, 3, "md_delim"), spazi)
        self.assertIn((3, "lineend", "md_h2"), spazi)

    def test_grassetto_evidenziato_e_delimitatori_elisi(self):
        # "testo **forte** ok": il grassetto copre "forte", i ** spariscono.
        spazi = self._tag_per_linea(_spazi_markdown(["testo **forte** ok"]))
        self.assertIn((8, 13, "md_grassetto"), spazi)
        self.assertEqual(
            [s for s in spazi if s[2] == "md_delim"], [(6, 8, "md_delim"), (13, 15, "md_delim")]
        )

    def test_blocco_codice_delimitato_da_fence(self):
        spazi = _spazi_markdown(["```", "print(1)", "```"])
        self.assertEqual(
            self._tag_per_linea(spazi, 1), [(0, "lineend", "md_codice")]
        )
        self.assertEqual(
            self._tag_per_linea(spazi, 0), [(0, "lineend", "md_delim")]
        )
        self.assertEqual(
            self._tag_per_linea(spazi, 2), [(0, "lineend", "md_delim")]
        )

    def test_codice_inline(self):
        spazi = self._tag_per_linea(_spazi_markdown(["digita `ls -l` ora"]))
        self.assertIn((8, 13, "md_codice"), spazi)

    def test_marker_elenco_puntato_e_numerato(self):
        spazi = self._tag_per_linea(_spazi_markdown(["  - voce"]))
        self.assertIn((2, 3, "md_marker"), spazi)
        spazi_numerati = self._tag_per_linea(_spazi_markdown(["3. voce"]))
        self.assertIn((0, 2, "md_marker"), spazi_numerati)

    def test_link_mostra_testo_ed_elide_url(self):
        # "vedi [sito](https://esempio.it).": visibile solo 'sito' con md_link.
        spazi = self._tag_per_linea(
            _spazi_markdown(["vedi [sito](https://esempio.it)."])
        )
        self.assertIn((6, 10, "md_link"), spazi)
        self.assertIn((10, 31, "md_delim"), spazi)

    def test_corsivo(self):
        spazi = self._tag_per_linea(_spazi_markdown(["un *tocco* di stile"]))
        self.assertIn((4, 9, "md_corsivo"), spazi)

    def test_colonna_tk_conta_le_emoji_astrali_come_due_unita(self):
        # Tk 8.x vede un'emoji astrale come una coppia di surrogati (2 colonne).
        self.assertEqual(_colonna_tk("🔍 ok", 0), 0)
        self.assertEqual(_colonna_tk("🔍 ok", 1), 2)
        self.assertEqual(_colonna_tk("🔍 ok", 4), 5)
        # Con Tcl 9 (una unità) non c'è correzione da applicare.
        self.assertEqual(_colonna_tk("🔍 ok", 4, unita_astrale=1), 4)
        # I caratteri del piano base non influiscono.
        self.assertEqual(_colonna_tk("àè ok", 5), 5)

    def test_spazi_rimangono_allineati_anche_dopo_un_emoji(self):
        # Regressione: «**Ricerca web**» dopo 🔍 veniva mostrato «Ricerca we».
        linea = "   - 🔍 **Ricerca web**: testo"
        # Vista come la conta Tk 8.x: ogni emoji astrale vale due colonne.
        vista_tk = "".join(c if ord(c) <= 0xFFFF else "§§" for c in linea)
        intervalli = {
            tag: (c1, c2)
            for _riga, c1, c2, tag in _spazi_markdown([linea])
            if tag != "md_delim"
        }
        c1, c2 = intervalli["md_grassetto"]
        testo = vista_tk[_colonna_tk(linea, c1): _colonna_tk(linea, c2)]
        self.assertEqual(testo, "Ricerca web")
        # Entrambi i delimitatori ** coprono esattamente gli asterischi.
        delimitatori = [
            vista_tk[_colonna_tk(linea, c1): _colonna_tk(linea, c2)]
            for _riga, c1, c2, tag in _spazi_markdown([linea])
            if tag == "md_delim"
        ]
        self.assertEqual(delimitatori, ["**", "**"])


class GuiProviderTests(unittest.TestCase):
    """La factory dei provider della GUI è testabile senza widget reali."""

    def setUp(self):
        self.gui = AgentGUI.__new__(AgentGUI)

    def test_lmstudio_crea_provider_con_url_configurato(self):
        provider = self.gui._crea_provider_da_config(
            "lmstudio",
            {"modello": "mia-rete", "base_url": "http://127.0.0.1:9999/v1"},
        )

        self.assertIsInstance(provider, LMStudioProvider)
        self.assertEqual(provider.model, "mia-rete")
        self.assertIn("127.0.0.1:9999", str(provider.client.base_url))

    def test_lmstudio_ricade_sull_url_predefinito(self):
        provider = self.gui._crea_provider_da_config(
            "lmstudio", {"modello": "mia-rete", "base_url": " "}
        )

        self.assertIn("localhost:1234", str(provider.client.base_url))

    def test_modello_vuoto_e_rifiutato_per_ogni_scheda(self):
        for scheda in ("lmstudio", "ollama", "openai"):
            with self.assertRaises(ValueError):
                self.gui._crea_provider_da_config(scheda, {"modello": "  "})

    def test_ollama_crea_provider_anche_senza_sdk(self):
        # Senza il pacchetto 'ollama' il provider ripiega sulla API HTTP:
        # la factory non deve più fallire.
        provider = self.gui._crea_provider_da_config(
            "ollama", {"modello": "qwen3:4b", "base_url": "http://ollama-lan:11434"}
        )

        self.assertIsInstance(provider, LLMProvider)
        self.assertEqual(provider.model, "qwen3:4b")
        self.assertEqual(provider.host, "http://ollama-lan:11434")

    def test_ollama_senza_host_ricade_sul_default(self):
        provider = self.gui._crea_provider_da_config(
            "ollama", {"modello": "qwen3:4b", "base_url": " "}
        )

        self.assertIsNone(provider.host)

    def test_ollama_complete_via_http_quando_manca_l_sdk(self):

        corpo = json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": "Fatto.",
                    "tool_calls": [
                        {"function": {"name": "list_files", "arguments": {"path": "/"}}}
                    ],
                }
            }
        ).encode()
        finto = mock.MagicMock()
        finto.read.return_value = corpo
        finto.__enter__.return_value = finto
        provider = OllamaProvider(model="qwen3:4b", host="http://ollama-lan:11434")
        with mock.patch("urllib.request.urlopen", return_value=finto) as aperta, \
                mock.patch("importlib.util.find_spec", return_value=None):
            esito = provider.complete([{"role": "user", "content": "ciao"}])

        self.assertEqual(esito.content, "Fatto.")
        self.assertEqual(esito.tool_calls[0].name, "list_files")
        self.assertEqual(esito.tool_calls[0].arguments, {"path": "/"})
        richiesta = aperta.call_args.args[0]
        self.assertEqual(richiesta.full_url, "http://ollama-lan:11434/api/chat")
        inviato = json.loads(richiesta.data.decode("utf-8"))
        self.assertEqual(inviato["model"], "qwen3:4b")
        self.assertFalse(inviato["stream"])
        self.assertNotIn("options", inviato)

    def test_ollama_http_inoltra_le_opzioni_extra(self):

        finto = mock.MagicMock()
        finto.read.return_value = json.dumps({"message": {"content": "ok"}}).encode()
        finto.__enter__.return_value = finto
        provider = OllamaProvider(model="qwen3:4b", temperature=0.2)
        with mock.patch("urllib.request.urlopen", return_value=finto) as aperta, \
                mock.patch("importlib.util.find_spec", return_value=None):
            provider.complete([{"role": "user", "content": "ciao"}])

        inviato = json.loads(aperta.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(inviato["options"], {"temperature": 0.2})

    def test_ollama_http_errore_di_connessione_leggibile(self):
        import urllib.error


        provider = OllamaProvider(model="qwen3:4b")
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ), mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "localhost:11434"):
                provider.complete([{"role": "user", "content": "ciao"}])

    def test_ollama_http_riprova_con_think_false_se_solo_thinking(self):

        pensiero = json.dumps(
            {"message": {"content": "", "thinking": "sto ragionando…"}}
        ).encode()
        risposta = json.dumps({"message": {"content": "Risposta vera."}}).encode()
        richieste = []

        def finta_urlopen(richiesta, timeout):
            finto = mock.MagicMock()
            payload = json.loads(richiesta.data.decode("utf-8"))
            richieste.append(payload)
            finto.read.return_value = (
                risposta if payload.get("think") is False else pensiero
            )
            finto.__enter__.return_value = finto
            return finto

        provider = OllamaProvider(model="kimi-k3:cloud")
        with mock.patch("urllib.request.urlopen", side_effect=finta_urlopen), \
                mock.patch("importlib.util.find_spec", return_value=None):
            esito = provider.complete([{"role": "user", "content": "chi sei?"}])

        self.assertEqual(esito.content, "Risposta vera.")
        self.assertEqual(len(richieste), 2)
        self.assertIs(richieste[1].get("think"), False)

    def test_ollama_http_non_ritenta_se_think_gia_impostato(self):

        pensiero = json.dumps(
            {"message": {"content": "", "thinking": "solo ragionamento"}}
        ).encode()
        richieste = []

        def finta_urlopen(richiesta, timeout):
            finto = mock.MagicMock()
            richieste.append(json.loads(richiesta.data.decode("utf-8")))
            finto.read.return_value = pensiero
            finto.__enter__.return_value = finto
            return finto

        provider = OllamaProvider(model="x", think=False)
        with mock.patch("urllib.request.urlopen", side_effect=finta_urlopen), \
                mock.patch("importlib.util.find_spec", return_value=None):
            esito = provider.complete([{"role": "user", "content": "ciao"}])

        self.assertEqual(len(richieste), 1)
        self.assertEqual(richieste[0]["options"], {"think": False})
        self.assertEqual(esito.content, "")

    def test_ollama_http_solleva_l_errore_dichiarato_dal_demone(self):

        corpo = json.dumps({"error": "model 'kimi-k3:cloud' not found"}).encode()
        finto = mock.MagicMock()
        finto.read.return_value = corpo
        finto.__enter__.return_value = finto
        provider = OllamaProvider(model="kimi-k3:cloud")
        with mock.patch("urllib.request.urlopen", return_value=finto), \
                mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                provider.complete([{"role": "user", "content": "ciao"}])

    def test_ollama_http_mostra_codice_e_dettaglio_http(self):
        import io
        import urllib.error


        provider = OllamaProvider(model="assente:cloud")
        errore = urllib.error.HTTPError(
            "http://localhost:11434/api/chat",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"error":"model not found"}'),
        )
        with mock.patch("urllib.request.urlopen", side_effect=errore), \
                mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                provider.complete([{"role": "user", "content": "ciao"}])

    def test_openai_senza_chiave_e_rifiutato(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                self.gui._crea_provider_da_config(
                    "openai", {"modello": "gpt-4o-mini", "api_key": "", "base_url": ""}
                )

    def test_openai_con_chiave_crea_provider(self):
        provider = self.gui._crea_provider_da_config(
            "openai",
            {"modello": "gpt-4o-mini", "api_key": "sk-test", "base_url": ""},
        )

        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.model, "gpt-4o-mini")

    def test_scheda_sconosciuta_e_rifiutata(self):
        with self.assertRaisesRegex(ValueError, "sconosciuto"):
            self.gui._crea_provider_da_config("huggingface", {"modello": "x"})

    def test_descrizione_provider_cita_modello_e_url(self):
        testo = self.gui._descrivi_provider(
            "lmstudio", {"modello": "mia-rete", "base_url": "http://h:1234/v1"}
        )
        self.assertIn("LM Studio", testo)
        self.assertIn("mia-rete", testo)
        self.assertIn("http://h:1234/v1", testo)

    def test_pulizia_rimuove_cluster_finale_con_e_senza_zwsp(self):
        from AI.providers import _ripulisci_testo_template as pulisci

        ZW = "​"
        sporco = (
            f" Sono Kimi, un modello di Moonshot AI. <{ZW}|close|> response "
            f"<{ZW}|sep|> <|close|> message <|sep|>"
        )
        pulito = pulisci(sporco)
        self.assertEqual(pulito, "Sono Kimi, un modello di Moonshot AI.")

    def test_pulizia_non_tocca_parole_legittime(self):
        from AI.providers import _ripulisci_testo_template as pulisci

        testo = "La response corretta arriva al message finale"
        self.assertEqual(pulisci(testo), testo)
        self.assertIsNone(pulisci(None))
        self.assertEqual(pulisci(""), "")

    def test_pulizia_conserva_uno_spazio_tra_parole(self):
        from AI.providers import _ripulisci_testo_template as pulisci

        self.assertEqual(pulisci("bla<|close|>fine"), "bla fine")
        self.assertEqual(pulisci("```py\nx = 1\n```\n<|close|>"), "```py\nx = 1\n```")

    def test_modelli_ollama_legge_la_risposta_api_tags(self):
        corpo = json.dumps(
            {"models": [{"name": "qwen3:4b"}, {"name": "glm-5.3-flash:cloud"}]}
        ).encode()
        finto = mock.MagicMock()
        finto.read.return_value = corpo
        finto.__enter__.return_value = finto
        with mock.patch("urllib.request.urlopen", return_value=finto):
            nomi = AgentGUI._modelli_ollama("http://localhost:11434")

        self.assertEqual(nomi, ["qwen3:4b", "glm-5.3-flash:cloud"])

    def test_modelli_ollama_ricade_sul_localhost_se_host_vuoto(self):
        finto = mock.MagicMock()
        finto.read.return_value = json.dumps({"models": []}).encode()
        finto.__enter__.return_value = finto
        with mock.patch("urllib.request.urlopen", return_value=finto) as aperta:
            AgentGUI._modelli_ollama("")

        richiesta = aperta.call_args.args[0]
        self.assertEqual(richiesta.full_url, "http://localhost:11434/api/tags")

    def test_modelli_ollama_errore_di_connessione_leggibile(self):
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with self.assertRaisesRegex(ValueError, "non raggiungibile"):
                AgentGUI._modelli_ollama("http://127.0.0.1:11434")

    def test_ollama_provider_ricorda_l_host_senza_importare_l_sdk(self):

        provider = OllamaProvider(model="qwen3:4b", host="http://192.168.1.4:11434")

        self.assertEqual(provider.model, "qwen3:4b")
        self.assertEqual(provider.host, "http://192.168.1.4:11434")


class TimelineRichiudibileTests(unittest.TestCase):
    def test_il_pulsante_apre_e_chiude_la_timeline(self):
        class FakeFrameGriglia:
            def __init__(self):
                self.visibile = False

            def grid(self, **_kw):
                self.visibile = True

            def grid_remove(self):
                self.visibile = False

        class FakePulsante:
            def __init__(self):
                self.testo = None

            def configure(self, **kw):
                self.testo = kw.get("text", self.testo)

        gui = AgentGUI.__new__(AgentGUI)
        gui.agent_frame = FakeFrameGriglia()
        gui.pulsante_timeline = FakePulsante()
        gui._timeline_visibile = False
        gui._timeline_layout = {"row": 4}

        gui._commuta_timeline()
        self.assertTrue(gui.agent_frame.visibile)
        self.assertEqual(gui.pulsante_timeline.testo, "▼ Timeline di esecuzione")

        gui._commuta_timeline()
        self.assertFalse(gui.agent_frame.visibile)
        self.assertEqual(gui.pulsante_timeline.testo, "▶ Timeline di esecuzione")


class PromptESandboxTests(unittest.TestCase):
    class FakePromptTesto:
        """Sostituto minimo di tk.Text per la logica del segnaposto."""

        def __init__(self):
            self.testo = ""
            self.tags = []

        def get(self, *_args, **_kw):
            if not self.testo:
                return "\n"
            return self.testo if self.testo.endswith("\n") else self.testo + "\n"

        def delete(self, *_args):
            self.testo = ""

        def insert(self, _pos, testo, *_args):
            self.testo += testo

        def tag_add(self, *args):
            self.tags.append(args)

        def tag_remove(self, *args):
            self.tags.append(("remove",) + args)

    def _gui_con_prompt(self):
        gui = AgentGUI.__new__(AgentGUI)
        gui.prompt = self.FakePromptTesto()
        gui._prompt_ha_placeholder = False
        return gui

    def test_segnaposto_mostra_suggerimento_e_toglie_alla_scrittura(self):
        gui = self._gui_con_prompt()

        gui._mostra_placeholder_prompt()
        self.assertTrue(gui._prompt_ha_placeholder)
        self.assertIn("Scrivi qui", gui.prompt.testo)
        self.assertEqual(gui._testo_prompt(), "")  # non è testo reale

        gui._rimuovi_placeholder_prompt()
        self.assertFalse(gui._prompt_ha_placeholder)
        self.assertEqual(gui.prompt.testo, "")

    def test_testo_reale_letto_e_placeholder_ripristinato_se_vuoto(self):
        gui = self._gui_con_prompt()

        gui.prompt.testo = "  una richiesta vera  "
        self.assertEqual(gui._testo_prompt(), "una richiesta vera")

        gui.prompt.testo = ""
        gui._aggiorna_placeholder_prompt()
        self.assertTrue(gui._prompt_ha_placeholder)

    def test_apri_sandbox_invoca_il_file_manager(self):
        gui = AgentGUI.__new__(AgentGUI)
        variabile = mock.MagicMock()
        variabile.get.return_value = "/tmp"
        gui.cartella_sandbox = variabile
        gui.messagebox = None
        with mock.patch.object(
            agente_gui_module.messagebox, "showerror", create=True
        ) as showerror, mock.patch(
            "subprocess.run", return_value=mock.MagicMock(returncode=0)
        ) as eseguito:
            gui._apri_sandbox()

        self.assertEqual(eseguito.call_count, 1)
        self.assertIn("/tmp", eseguito.call_args.args[0])
        self.assertFalse(showerror.called)

    def test_apri_sandbox_con_cartella_assente_avvisa(self):
        gui = AgentGUI.__new__(AgentGUI)
        variabile = mock.MagicMock()
        variabile.get.return_value = "/percorso/che/non/esiste/davvero"
        gui.cartella_sandbox = variabile
        with mock.patch.object(
            agente_gui_module.messagebox, "showerror", create=True
        ) as showerror:
            gui._apri_sandbox()

        self.assertTrue(showerror.called)


if __name__ == "__main__":
    unittest.main()
