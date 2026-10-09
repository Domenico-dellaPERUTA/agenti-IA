import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from AI import AI, LLMProvider, LLMResponse


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


def make_application(sandbox, *, provider=None, initial_prompt="test prompt",
                     base_system_prompt="test system"):
    return AI(
        provider or DummyProvider(),
        sandbox,
        initial_prompt=initial_prompt,
        base_system_prompt=base_system_prompt,
    )


class ApplicationOrchestrationTests(unittest.TestCase):
    def test_factory_builds_independent_agents_with_shared_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory)
            with patch(
                "AI.application.InternetAccess", return_value=DummyInternet()
            ):
                provider = DummyProvider()
                application = make_application(sandbox, provider=provider)
                orchestrator = application.create_orchestrator()
                planner = orchestrator.agent_factory("prompt planner")
                worker = orchestrator.agent_factory("prompt worker")

        self.assertIsNot(planner, worker)
        self.assertIs(planner.provider, provider)
        self.assertIs(worker.provider, provider)
        self.assertEqual(planner.sandbox, worker.sandbox)
        self.assertNotEqual(planner.messages, worker.messages)
        self.assertIn("web_search", planner.tools)
        self.assertIn("read_webpage", worker.tools)

    def test_factory_uses_provider_and_configuration_from_caller(self):
        provider = DummyProvider()
        with tempfile.TemporaryDirectory() as directory, patch(
            "AI.application.InternetAccess", return_value=DummyInternet()
        ):
            application = make_application(
                Path(directory),
                provider=provider,
                initial_prompt="caller prompt",
                base_system_prompt="caller system prompt",
            )
            agent = application.create_agent()

        self.assertIs(agent.provider, provider)
        self.assertEqual(application.initial_prompt, "caller prompt")
        self.assertTrue(agent.messages[0]["content"].startswith("caller system prompt"))
        self.assertIn(
            "non è disponibile uno strumento di esecuzione script",
            agent.messages[0]["content"],
        )


if __name__ == "__main__":
    unittest.main()
