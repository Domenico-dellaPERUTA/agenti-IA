import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from AI import LLMProvider, LLMResponse
from agente import crea_orchestratore


class DummyProvider(LLMProvider):
    def complete(self, messages, tools=None):
        return LLMResponse(content="Risposta simulata")


class DummyInternet:
    def web_search(self, query):
        return query

    def read_webpage(self, url):
        return url

    def sources(self):
        return []


class ApplicationOrchestrationTests(unittest.TestCase):
    def test_factory_builds_independent_agents_with_shared_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory)
            with (
                patch(
                    "agente.LMStudioProvider",
                    side_effect=lambda **_kwargs: DummyProvider(),
                ) as provider_factory,
                patch("agente.InternetAccess", return_value=DummyInternet()),
            ):
                orchestrator = crea_orchestratore(sandbox)
                planner = orchestrator.agent_factory("prompt planner")
                worker = orchestrator.agent_factory("prompt worker")

        self.assertIsNot(planner, worker)
        self.assertIsNot(planner.provider, worker.provider)
        self.assertEqual(planner.sandbox, worker.sandbox)
        self.assertNotEqual(planner.messages, worker.messages)
        self.assertIn("web_search", planner.tools)
        self.assertIn("read_webpage", worker.tools)
        self.assertEqual(provider_factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
