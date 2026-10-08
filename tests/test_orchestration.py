import json
import threading
import time
import unittest

from AI import AgentOrchestrator, LLMResponse, OrchestrationError


class FakeAgent:
    def __init__(self, system_prompt, state):
        self.system_prompt = system_prompt
        self.state = state
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tools = {"unsafe_tool": lambda: None}
        self.allowed_tools = None
        self.direct_runs = 0
        self.writer_runs = 0

    def restrict_tools(self, allowed_tools):
        if self.allowed_tools is not None:
            allowed_tools = self.allowed_tools & allowed_tools
        self.allowed_tools = frozenset(allowed_tools)
        self.tools = {
            name: tool for name, tool in self.tools.items()
            if name in self.allowed_tools
        }

    def send(self, message):
        self.messages.append({"role": "user", "content": message})

    def run(self, on_response=None, on_tool_result=None):
        if "pianificatore" in self.system_prompt:
            answer = self.state["plan"]
        elif "sintesi" in self.system_prompt:
            self.state["syntheses"] += 1
            self.state["phase_order"].append("synthesis")
            self.state["synthesis_prompt"] = self.messages[-1]["content"]
            answer = self.state.get("synthesis_answer", "Risposta sintetica")
        elif "agente di ricerca web" in self.system_prompt:
            self.state["phase_order"].append("research")
            answer = self.state.get("research_answer", "Risultati verificabili e fonti")
        elif "agente dedicato esclusivamente alla creazione" in self.system_prompt:
            self.writer_runs += 1
            self.state["writer_runs"] += 1
            self.state["phase_order"].append("writer")
            if (
                self.writer_runs > 1
                and self.state.get("create_file_on_retry")
                and on_tool_result is not None
            ):
                on_tool_result("create_file", "Creato: risultati.txt")
            answer = self.state.get("writer_answer", "Il file risultati.txt è stato creato.")
        elif "Completa tutte le azioni esplicitamente richieste" in self.system_prompt:
            self.state["direct_runs"] += 1
            self.direct_runs += 1
            if (
                self.direct_runs > 1
                and self.state.get("create_file_on_retry")
                and on_tool_result is not None
            ):
                on_tool_result("create_file", "Creato: risultati.txt")
            answer = self.state.get(
                "direct_answer",
                "Il file risultati.txt è stato creato."
                if self.direct_runs > 1 and self.state.get("create_file_on_retry")
                else "Risposta diretta",
            )
        else:
            prompt = self.messages[-1]["content"]
            if "fallisci" in prompt:
                raise RuntimeError("errore controllato del worker")
            with self.state["lock"]:
                self.state["active"] += 1
                self.state["max_active"] = max(
                    self.state["max_active"], self.state["active"]
                )
            try:
                if "lento" in prompt:
                    time.sleep(0.05)
                elif "terzo" in prompt:
                    time.sleep(0.1)
                else:
                    time.sleep(0.005)
                answer = prompt.split("Istruzioni:\n", 1)[1]
            finally:
                with self.state["lock"]:
                    self.state["active"] -= 1

        if on_response is not None:
            on_response(LLMResponse(content=answer))
        return answer


class OrchestrationTests(unittest.TestCase):
    def make_orchestrator(self, plan, **kwargs):
        state = {
            "plan": plan,
            "agents": [],
            "active": 0,
            "max_active": 0,
            "lock": threading.Lock(),
            "direct_runs": 0,
            "create_file_on_retry": False,
            "writer_runs": 0,
            "syntheses": 0,
            "phase_order": [],
        }

        def factory(system_prompt):
            agent = FakeAgent(system_prompt, state)
            state["agents"].append(agent)
            return agent

        return AgentOrchestrator(factory, **kwargs), state

    @staticmethod
    def parallel_plan(*tasks):
        return json.dumps({"mode": "parallel", "tasks": list(tasks)})

    @staticmethod
    def task(task_id, instruction, title=None):
        return {
            "id": task_id,
            "title": title or task_id.title(),
            "instructions": instruction,
        }

    def test_direct_plan_runs_one_normal_agent_without_workers(self):
        orchestrator, state = self.make_orchestrator('{"mode":"direct","tasks":[]}')
        events = []

        result = orchestrator.run("Rispondi direttamente", on_event=events.append)

        self.assertEqual(result.mode, "direct")
        self.assertEqual(result.answer, "Risposta diretta")
        self.assertEqual(result.tasks, [])
        self.assertEqual(state["direct_runs"], 1)
        self.assertEqual(state["syntheses"], 0)
        self.assertEqual(state["agents"][0].allowed_tools, frozenset())
        direct_agent = next(
            agent
            for agent in state["agents"]
            if "Completa tutte le azioni esplicitamente richieste"
            in agent.system_prompt
        )
        self.assertIn("non fermarti a una risposta parziale", direct_agent.system_prompt)
        self.assertIn("usa create_file", direct_agent.system_prompt)
        self.assertIn("non conclude il lavoro", direct_agent.system_prompt)
        self.assertEqual(
            [
                (event["agent_id"], event["type"])
                for event in events
                if event["type"].startswith("agent_")
            ],
            [
                ("planner", "agent_started"),
                ("planner", "agent_completed"),
                ("direct", "agent_started"),
                ("direct", "agent_completed"),
            ],
        )

    def test_direct_plan_with_tasks_warns_and_uses_direct_agent(self):
        plan = json.dumps(
            {
                "mode": "direct",
                "tasks": [
                    self.task("research", "Cerca fonti pubbliche"),
                    self.task("write", "Crea il file richiesto"),
                ],
            }
        )
        orchestrator, state = self.make_orchestrator(plan)
        events = []

        result = orchestrator.run("Cerca informazioni e riassumi le fonti", on_event=events.append)

        self.assertEqual(result.mode, "direct")
        self.assertEqual(result.answer, "Risposta diretta")
        self.assertEqual(state["direct_runs"], 1)
        self.assertEqual(state["syntheses"], 0)
        self.assertEqual(len(state["agents"]), 2)
        warning = next(event for event in events if event["type"] == "plan_warning")
        self.assertIn("2 task", warning["message"])
        self.assertIn("saranno ignorati", warning["message"])

    def test_requested_file_gets_one_retry_and_requires_successful_tool(self):
        orchestrator, state = self.make_orchestrator('{"mode":"direct","tasks":[]}')
        state["create_file_on_retry"] = True
        state["writer_answer"] = "Il file non è stato creato."
        events = []

        result = orchestrator.run(
            "Cerca annunci e creami un file txt con i risultati",
            on_event=events.append,
        )

        self.assertTrue(result.answer.endswith("File creato: risultati.txt"))
        self.assertNotIn("non è stato creato", result.answer)
        self.assertEqual(state["direct_runs"], 0)
        self.assertEqual(state["writer_runs"], 2)
        self.assertEqual(state["phase_order"], ["research", "writer", "writer"])
        researcher, writer = state["agents"][1:]
        self.assertEqual(researcher.allowed_tools, frozenset({"web_search", "read_webpage"}))
        self.assertEqual(writer.allowed_tools, frozenset({"create_file"}))
        self.assertIn(
            ("Agente di creazione file", "Creato: risultati.txt"),
            [
                (event.get("title"), event.get("message"))
                for event in events
                if event["type"] == "tool_result"
            ],
        )
        self.assertTrue(any(event["type"] == "agent_retry" for event in events))
        self.assertIn("create_file", writer.messages[-1]["content"])
        self.assertNotIn(
            "Il file non è stato creato.",
            [event.get("message") for event in events],
        )

    def test_requested_file_is_not_reported_as_created_without_tool_success(self):
        orchestrator, state = self.make_orchestrator('{"mode":"direct","tasks":[]}')
        state["writer_answer"] = "Il file richiesti.txt è stato creato."
        events = []

        with self.assertRaisesRegex(OrchestrationError, "non ha verificato"):
            orchestrator.run(
                "Cerca e creami un file con i risultati",
                on_event=events.append,
            )

        self.assertEqual(state["direct_runs"], 0)
        self.assertEqual(state["writer_runs"], 2)
        self.assertNotIn(
            "Il file richiesti.txt è stato creato.",
            [event.get("message") for event in events],
        )
        self.assertIn(
            ("file_writer", "agent_failed"),
            [
                (event.get("agent_id"), event["type"])
                for event in events
                if event["type"].startswith("agent_")
            ],
        )
        self.assertNotIn(
            ("file_writer", "agent_completed"),
            [
                (event.get("agent_id"), event["type"])
                for event in events
                if event["type"].startswith("agent_")
            ],
        )

    def test_parallel_file_writer_starts_after_all_research_and_synthesis(self):
        orchestrator, state = self.make_orchestrator(
            self.parallel_plan(
                self.task("first", "risultato prima fonte"),
                self.task("second", "risultato seconda fonte"),
            )
        )
        state["create_file_on_retry"] = True
        events = []

        result = orchestrator.run(
            "Fai una ricerca e crea un file txt con i risultati",
            on_event=events.append,
        )

        self.assertTrue(result.answer.endswith("File creato: risultati.txt"))
        self.assertEqual(state["phase_order"][-3:], ["synthesis", "writer", "writer"])
        self.assertEqual(state["writer_runs"], 2)
        self.assertEqual(
            [agent.allowed_tools for agent in state["agents"][1:3]],
            [frozenset({"web_search", "read_webpage"})] * 2,
        )
        completed_positions = [
            index for index, event in enumerate(events)
            if event["type"] == "task_completed"
        ]
        writer_started = next(
            index for index, event in enumerate(events)
            if event.get("agent_id") == "file_writer"
            and event["type"] == "agent_started"
        )
        self.assertEqual(len(completed_positions), 2)
        self.assertTrue(all(index < writer_started for index in completed_positions))
        self.assertEqual(state["agents"][-1].allowed_tools, frozenset({"create_file"}))

    def test_parallel_results_follow_plan_order_and_worker_limit(self):
        orchestrator, state = self.make_orchestrator(
            self.parallel_plan(
                self.task("slow", "lento risultato"),
                {
                    **self.task("fast", "risultato veloce"),
                    "context": "contesto mirato",
                },
                self.task("third", "terzo risultato"),
            ),
            max_workers=2,
        )
        events = []

        result = orchestrator.run("Analizza tre aspetti", on_event=events.append)

        self.assertEqual(result.mode, "parallel")
        self.assertEqual([item.task_id for item in result.tasks], ["slow", "fast", "third"])
        self.assertEqual([item.status for item in result.tasks], ["completed"] * 3)
        self.assertLessEqual(state["max_active"], 2)
        self.assertEqual(state["max_active"], 2)
        self.assertEqual(result.answer, "Risposta sintetica")
        self.assertEqual(state["syntheses"], 1)
        self.assertEqual(state["agents"][0].allowed_tools, frozenset())
        worker_agents = [
            agent
            for agent in state["agents"]
            if "pianificatore" not in agent.system_prompt
            and "sintesi" not in agent.system_prompt
        ]
        self.assertEqual(len(worker_agents), 3)
        self.assertTrue(all(agent.allowed_tools for agent in worker_agents))
        self.assertTrue(
            all(
                "Analizza tre aspetti" not in agent.messages[-1]["content"]
                for agent in worker_agents
            )
        )
        self.assertTrue(
            any("contesto mirato" in agent.messages[-1]["content"] for agent in worker_agents)
        )
        completed_ids = [
            event["task_id"]
            for event in events
            if event["type"] == "task_completed"
        ]
        self.assertEqual(completed_ids, ["fast", "slow", "third"])
        self.assertIn(
            ("synthesizer", "agent_completed"),
            [
                (event.get("agent_id"), event["type"])
                for event in events
                if event["type"].startswith("agent_")
            ],
        )

    def test_partial_worker_failure_is_reported_and_synthesized(self):
        orchestrator, state = self.make_orchestrator(
            self.parallel_plan(
                self.task("ok", "risultato buono"),
                self.task("broken", "fallisci"),
            )
        )
        events = []

        result = orchestrator.run("Analizza", on_event=events.append)

        self.assertEqual([item.status for item in result.tasks], ["completed", "failed"])
        self.assertEqual(result.tasks[1].error, "errore controllato del worker")
        self.assertEqual(state["syntheses"], 1)
        self.assertIn("broken", state["synthesis_prompt"])
        self.assertTrue(any(event["type"] == "task_failed" for event in events))

    def test_all_worker_failures_skip_synthesis(self):
        orchestrator, state = self.make_orchestrator(
            self.parallel_plan(
                self.task("one", "fallisci"),
                self.task("two", "fallisci"),
            )
        )

        result = orchestrator.run("Analizza")

        self.assertIsNone(result.answer)
        self.assertEqual([item.status for item in result.tasks], ["failed", "failed"])
        self.assertEqual(state["syntheses"], 0)

    def test_invalid_plan_fails_before_workers_start(self):
        orchestrator, state = self.make_orchestrator(
            self.parallel_plan(
                self.task("duplicate", "primo"),
                self.task("duplicate", "secondo"),
            )
        )

        with self.assertRaisesRegex(OrchestrationError, "duplicato"):
            orchestrator.run("Analizza")

        self.assertEqual(len(state["agents"]), 1)

    def test_rejects_malformed_json_and_limits(self):
        malformed, malformed_state = self.make_orchestrator("```json\n{}\n```")
        with self.assertRaisesRegex(OrchestrationError, "JSON non valido"):
            malformed.run("Analizza")
        self.assertEqual(len(malformed_state["agents"]), 1)

        too_many, _ = self.make_orchestrator(
            self.parallel_plan(
                *(self.task(f"task-{index}", "istruzioni") for index in range(6))
            )
        )
        with self.assertRaisesRegex(OrchestrationError, "da 2 a 5"):
            too_many.run("Analizza")

    def test_validates_worker_configuration(self):
        with self.assertRaisesRegex(ValueError, "tra 1 e 4"):
            self.make_orchestrator('{"mode":"direct","tasks":[]}', max_workers=5)
        with self.assertRaisesRegex(ValueError, "tra 2 e 5"):
            self.make_orchestrator('{"mode":"direct","tasks":[]}', max_tasks=1)


if __name__ == "__main__":
    unittest.main()
