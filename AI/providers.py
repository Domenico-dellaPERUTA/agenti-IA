"""Adattatori che traducono il protocollo comune dell'agente nei provider LLM.

Ogni adattatore riceve gli stessi messaggi e callable, converte i callable
nello schema tool richiesto dal servizio e converte la risposta del servizio
in ``LLMResponse``. Gli SDK opzionali vengono importati solo quando il relativo
provider viene effettivamente usato.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from .core import LLMProvider, LLMResponse, ToolCall

# -------------------------------------------------------------------#
#                           OllamaProvider                           #  
#--------------------------------------------------------------------#

class OllamaProvider(LLMProvider):
    """Provider che invoca i modelli Ollama via SDK locale.

    Converte gli strumenti Python in schemi JSON compatibili con Ollama e
    riconverte le tool call ricevute in oggetti ``ToolCall`` del progetto.

    Esempio:
        >>> from IA import Agent, OllamaProvider
        >>> provider = OllamaProvider(model="qwen3:4b")
        >>> agent = Agent(provider)
        >>> agent.send("Ciao!")
        >>> risposta = agent.run()
    """

    def __init__(self, model: str, **kwargs):
        """Memorizza il modello Ollama e le opzioni da inoltrare all'SDK."""
        self.model = model
        self.kwargs = kwargs

    def complete(self, messages: list[dict], tools: list[Callable] | None = None) -> LLMResponse:
        """Invia messaggi e strumenti all'istanza Ollama locale selezionata.

        L'importazione differita mantiene Ollama una dipendenza opzionale.
        La risposta SDK viene normalizzata nell'oggetto comune del progetto.
        """
        from ollama import chat

        tool_specs = None
        if tools:
            tool_specs = [self._tool_to_schema(tool) for tool in tools]

        response = chat(model=self.model, messages=messages, tools=tool_specs, **self.kwargs)
        tool_calls = []

        for call in getattr(response.message, "tool_calls", []) or []:
            tool_calls.append(
                ToolCall(
                    name=call.function.name,
                    arguments=dict(call.function.arguments or {}),
                )
            )

        return LLMResponse(
            content=getattr(response.message, "content", None),
            tool_calls=tool_calls,
        )

    def _tool_to_schema(self, fn: Callable) -> dict[str, Any]:
        """Traduce firma, tipi e docstring Python nello schema tool Ollama."""
        signature = inspect.signature(fn, eval_str=True)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            annotation = param.annotation
            if annotation is inspect._empty:
                json_type = "string"
            else:
                json_type = self._annotation_to_json_type(annotation)

            properties[name] = {"type": json_type}
            if param.default is inspect._empty:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @staticmethod
    def _annotation_to_json_type(annotation):
        """Mappa i tipi Python supportati sui tipi JSON dello schema tool."""
        if annotation in (str,):
            return "string"
        if annotation in (int, float):
            return "number"
        if annotation is bool:
            return "boolean"
        if annotation in (list, tuple, set):
            return "array"
        if annotation is dict:
            return "object"
        return "string"

# -------------------------------------------------------------------#
#                           OpenAIProvider.                          #  
#--------------------------------------------------------------------#

class OpenAIProvider(LLMProvider):
    """Provider basato sulla API Chat Completions di OpenAI.

    Adatta le funzioni Python agli strumenti supportati da OpenAI e
    trasforma le tool call in oggetti del sistema per essere eseguite
    dall'``Agent``.

    Esempio con un client OpenAI:
        >>> from openai import OpenAI
        >>> client = OpenAI(api_key="la-tua-api-key")
        >>> provider = OpenAIProvider(model="gpt-4o-mini", client=client)

    In alternativa, il client viene creato automaticamente:
        >>> provider = OpenAIProvider(
        ...     model="gpt-4o-mini",
        ...     api_key="la-tua-api-key",
        ... )
    """

    def __init__(
        self,
        model: str,
        client: Any | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """Prepara un client OpenAI o compatibile con OpenAI.

        Se ``client`` è fornito viene riutilizzato (utile per test e configurazioni
        personalizzate); altrimenti il client SDK viene creato con URL e chiave.
        """
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)

        self.model = model
        self.client = client

    def complete(self, messages: list[dict], tools: list[Callable] | None = None) -> LLMResponse:
        """Chiama Chat Completions e converte testo e tool call nel formato comune.

        In caso di errore di connessione, aggiunge l'URL configurato al messaggio
        dell'eccezione per aiutare a correggere l'indirizzo del provider.
        """
        tool_specs = None
        if tools:
            tool_specs = [self._tool_to_schema(tool) for tool in tools]

        from openai import APIConnectionError

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_specs,
            )
        except APIConnectionError as error:
            base_url = str(self.client.base_url).rstrip("/")
            raise RuntimeError(
                f"Impossibile raggiungere il provider LLM all'indirizzo {base_url}. "
                "Verifica l'indirizzo e che il provider sia disponibile."
            ) from error

        message = response.choices[0].message
        tool_calls = []
        for call in getattr(message, "tool_calls", []) or []:
            arguments = call.function.arguments
            try:
                parsed_arguments = json.loads(arguments)
            except (TypeError, ValueError):
                parsed_arguments = {}

            tool_calls.append(
                ToolCall(
                    name=call.function.name,
                    arguments=parsed_arguments,
                )
            )

        return LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
        )

    def _tool_to_schema(self, fn: Callable) -> dict[str, Any]:
        """Costruisce lo schema funzione OpenAI leggendo firma e docstring."""
        signature = inspect.signature(fn, eval_str=True)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            annotation = param.annotation
            if annotation is inspect._empty:
                json_type = "string"
            else:
                json_type = self._annotation_to_json_type(annotation)

            properties[name] = {"type": json_type}
            if param.default is inspect._empty:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @staticmethod
    def _annotation_to_json_type(annotation):
        """Traduce annotazioni Python semplici nei tipi JSON di OpenAI."""
        if annotation in (str,):
            return "string"
        if annotation in (int, float):
            return "number"
        if annotation is bool:
            return "boolean"
        if annotation in (list, tuple, set):
            return "array"
        if annotation is dict:
            return "object"
        return "string"


# -------------------------------------------------------------------#
#                           LMStudioProvider                         #  
#--------------------------------------------------------------------#

class LMStudioProvider(OpenAIProvider):
    """Provider per eseguire modelli caricati in LM Studio.

    LM Studio espone un endpoint compatibile con le API OpenAI, quindi
    questo provider delega tutto al comportamento di ``OpenAIProvider``
    usando un client configurato per il server locale di LM Studio.

    Esempio:
        >>> from IA import Agent, LMStudioProvider
        >>> provider = LMStudioProvider(model="nome-del-modello")
        >>> agent = Agent(provider, system_prompt="Sei un assistente utile.")
        >>> agent.send("Ciao!")
        >>> risposta = agent.run()

    L'endpoint predefinito è ``http://localhost:1234/v1``. È possibile
    sostituirlo quando LM Studio ascolta su un indirizzo diverso:
        >>> provider = LMStudioProvider(
        ...     model="nome-del-modello",
        ...     base_url="http://localhost:1234/v1",
        ... )
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        client: Any | None = None,
    ):
        """Configura l'adattatore OpenAI per il server locale di LM Studio.

        ``api_key`` è impostata a un segnaposto perché l'endpoint locale
        compatibile normalmente non richiede credenziali.
        """
        super().__init__(
            model=model,
            client=client,
            base_url=base_url,
            api_key="not-needed",
        )
