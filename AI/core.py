from __future__ import annotations

from dataclasses import dataclass, field
import json
from collections.abc import Callable
from typing import Any


@dataclass
class ToolCall:
    """Rappresenta una chiamata a una funzione generata dal modello.

    Gli strumenti di un modello possono restituire più azioni da eseguire,
    ciascuna identificata dal nome e dai parametri da passare.

    Esempio:
        >>> chiamata = ToolCall(
        ...     name="somma",
        ...     arguments={"a": 2, "b": 3},
        ... )
    """

    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Contiene la risposta di un provider LLM.

    Una risposta può includere sia un testo generato dal modello sia una
    lista di tool call da eseguire in modo automatico.

    Esempio:
        >>> risposta = LLMResponse(content="Ciao!")
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider:
    """Interfaccia astratta per i provider di modelli linguistici.

    Ogni provider specifico deve implementare il metodo ``complete`` per
    inviare messaggi al modello e trasformare la risposta in un
    ``LLMResponse``.

    Esempio:
        >>> class MioProvider(LLMProvider):
        ...     def complete(self, messages, tools=None):
        ...         return LLMResponse(content="Risposta locale")
        >>> provider = MioProvider()
    """

    def complete(self, messages: list[dict], tools: list[Callable] | None = None) -> LLMResponse:
        raise NotImplementedError("Il provider deve implementare complete().")


class Agent:
    """Gestisce la conversazione e l'esecuzione di tool del modello.

    L'agent mantiene lo stato della chat, registra gli strumenti disponibili e
    coordina l'esecuzione delle chiamate generate dal modello.

    Esempio:
        >>> provider = MioProvider()
        >>> agent = Agent(provider, system_prompt="Sei utile.")
        >>> agent.send("Ciao!")
        >>> risposta = agent.run()
    """

    def __init__(self, provider: LLMProvider, system_prompt: str | None = None):
        self.provider = provider
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.tools: dict[str, Callable] = {}

    def add_tool(self, fn: Callable):
        self.tools[fn.__name__] = fn

    def send(self, message: str, role: str = "user"):
        self.messages.append({"role": role, "content": message})

    def _stringify_tool_result(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)
        return json.dumps(value, ensure_ascii=False)

    def run(self):
        while True:
            response = self.provider.complete(self.messages, list(self.tools.values()))

            if response.content is not None and response.content != "":
                self.messages.append({"role": "assistant", "content": response.content})

            if not response.tool_calls:
                return response.content

            for call in response.tool_calls:
                tool_fn = self.tools.get(call.name)
                if tool_fn is None:
                    raise ValueError(f"Tool non registrata: {call.name}")

                risultato = tool_fn(**call.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_name": call.name,
                    "content": self._stringify_tool_result(risultato),
                })
