import json
import threading
import time
import unittest

from AI import AgenteOrchestratore, LLMResponse, OrchestrationError


class FakeAgent:
    def __init__(self, system_prompt, state):
        self.system_prompt = system_prompt
        self.state = state
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tools = {"unsafe_tool": lambda: None}
        self.allowed_tools = None

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
            self.state["synthesis_prompt"] = self.messages[-1]["content"]
            answer = self.state.get("synthesis_answer", "Risposta sintetica")
        elif "Completa tutte le azioni esplicitamente richieste" in self.system_prompt:
            self.state["direct_runs"] += 1
            answer = self.state.get("direct_answer", "Risposta diretta")
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
            "syntheses": 0,
        }

        def factory(system_prompt):
            agent = FakeAgent(system_prompt, state)
            state["agents"].append(agent)
            return agent

        return AgenteOrchestratore(factory, **kwargs), state

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

        result = orchestrator.run("Cerca informazioni e crea un file", on_event=events.append)

        self.assertEqual(result.mode, "direct")
        self.assertEqual(result.answer, "Risposta diretta")
        self.assertEqual(state["direct_runs"], 1)
        self.assertEqual(state["syntheses"], 0)
        self.assertEqual(len(state["agents"]), 2)
        warning = next(event for event in events if event["type"] == "plan_warning")
        self.assertIn("2 task", warning["message"])
        self.assertIn("saranno ignorati", warning["message"])

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
