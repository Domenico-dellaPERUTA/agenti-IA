"""Adattatori che traducono il protocollo comune dell'agente nei provider LLM.

Ogni adattatore riceve gli stessi messaggi e callable, converte i callable
nello schema tool richiesto dal servizio e converte la risposta del servizio
in ``LLMResponse``. Gli SDK opzionali vengono importati solo quando il relativo
provider viene effettivamente usato.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .core import LLMProvider, LLMResponse, ToolCall

# Un "cluster" di impalcatura del template chat inizia sempre con un token di
# controllo ``<|...|>`` e può proseguire con altri token o con le
# parole-ponte che i modelli lasciano accanto (response/message/sep).
_CLUSTER_TEMPLATE_CHAT = re.compile(
    r"[ \t]*<(?:​)?\|[^|]*(?:​)?\|>"
    r"(?:\s+(?:<(?:​)?\|[^|]*(?:​)?\|>|(?:response|message|sep)\b))*"
)


def _ripulisci_testo_template(testo: str | None) -> str | None:
    """Rimuove i residui dei template chat che alcuni modelli lasciano nel testo.

    Alcuni modelli (per esempio i cloud Ollama con template dedicati) lasciano
    nel contenuto sequenze come ``<|close|> response <|sep|> <|close|>
    message <|sep|>``: sono impalcatura del formato conversazionale, non testo
    destinato all'utente. Un cluster viene rimosso solo se inizia con un
    token di controllo, così un testo legittimo che contiene le parole
    "response"/"message"/"sep" non viene toccato; quando il cluster separa due
    parole viene sostituito da un singolo spazio.
    """
    if not isinstance(testo, str) or ("<|" not in testo and "<​|" not in testo):
        return testo

    def _togli_cluster(match: re.Match) -> str:
        inizio, fine = match.span()
        prima = testo[inizio - 1] if inizio > 0 else ""
        dopo = testo[fine] if fine < len(testo) else ""
        if prima and dopo and not prima.isspace() and not dopo.isspace():
            return " "
        return ""

    return _CLUSTER_TEMPLATE_CHAT.sub(_togli_cluster, testo).strip()

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

    def __init__(self, model: str, host: str | None = None, **kwargs):
        """Memorizza il modello Ollama e le opzioni da inoltrare all'SDK.

        Args:
            model: Nome del modello Ollama (per esempio ``qwen3:4b``).
            host: Indirizzo del demone Ollama; con ``None`` si usa il default
                dell'SDK (il demone locale, tipicamente http://localhost:11434).
            kwargs: Opzioni aggiuntive passate a ``chat()``.
        """
        self.model = model
        self.host = host
        self.kwargs = kwargs

    def complete(self, messages: list[dict], tools: list[Callable] | None = None) -> LLMResponse:
        """Invia messaggi e strumenti all'istanza Ollama selezionata.

        Se il pacchetto ``ollama`` è installato si usa l'SDK Python, che
        quando ``host`` è ``None`` segue anche la variabile d'ambiente
        OLLAMA_HOST; in caso contrario si ripiega sulla API HTTP del demone
        con la sola libreria standard, così il provider funziona comunque
        anche senza dipendenze facoltative.

        La risposta viene sempre normalizzata nell'oggetto comune del progetto.
        """
        tool_specs = None
        if tools:
            tool_specs = [self._tool_to_schema(tool) for tool in tools]

        if importlib.util.find_spec("ollama") is not None:
            return self._complete_via_sdk(messages, tool_specs)
        return self._complete_via_http(messages, tool_specs)

    def _complete_via_sdk(self, messages: list[dict], tool_specs: list[dict] | None) -> LLMResponse:
        """Invoca l'SDK Python ollama e normalizza la risposta.

        Come nel ramo HTTP, una risposta che riempie solo ``thinking`` viene
        ritentata una volta con ``think=False``; se l'SDK installato non
        supporta il parametro si rinuncia senza fallire.
        """

        def chiama(**extra: Any):
            argomenti = {
                "model": self.model,
                "messages": messages,
                "tools": tool_specs,
            }
            argomenti.update(self.kwargs)
            argomenti.update(extra)
            if self.host:
                from ollama import Client

                return Client(host=self.host).chat(**argomenti)
            from ollama import chat

            return chat(**argomenti)

        response = chiama()
        contenuto = getattr(response.message, "content", None)
        if (
            not (isinstance(contenuto, str) and contenuto.strip())
            and getattr(response.message, "thinking", None)
            and "think" not in self.kwargs
        ):
            try:
                response = chiama(think=False)
            except TypeError:
                pass  # SDK troppo vecchio per il parametro think
        tool_calls = []

        for call in getattr(response.message, "tool_calls", []) or []:
            tool_calls.append(
                ToolCall(
                    name=call.function.name,
                    arguments=dict(call.function.arguments or {}),
                )
            )

        return LLMResponse(
            content=_ripulisci_testo_template(getattr(response.message, "content", None)),
            tool_calls=tool_calls,
        )

    def _complete_via_http(self, messages: list[dict], tool_specs: list[dict] | None) -> LLMResponse:
        """Invoca ``POST /api/chat`` del demone Ollama con la libreria standard.

        Usata quando l'SDK non è installato: le opzioni extra di ``complete``
        (``kwargs``) vengono inoltrate come campo ``options`` della richiesta.
        I modelli "thinking" (per esempio i cloud kimi/deepseek) possono
        riempire il campo ``thinking`` lasciando ``content`` vuoto: in quel
        caso si riprova una sola volta con ``think: false``, che chiede al
        demone di produrre direttamente la risposta.
        """
        host = (self.host or "http://localhost:11434").rstrip("/")
        corpo: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tool_specs:
            corpo["tools"] = tool_specs
        if self.kwargs:
            corpo["options"] = dict(self.kwargs)

        def invia(payload: dict[str, Any]) -> dict[str, Any]:
            richiesta = urllib.request.Request(
                f"{host}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(richiesta, timeout=120) as risposta:
                    dati = json.loads(risposta.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                dettaglio = error.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(
                    f"Il server Ollama all'indirizzo {host} ha risposto "
                    f"HTTP {error.code}: {dettaglio}"
                ) from error
            except (urllib.error.URLError, OSError) as error:
                motivo = getattr(error, "reason", error)
                raise RuntimeError(
                    f"Impossibile raggiungere il provider Ollama all'indirizzo {host} "
                    f"({motivo}). Verifica l'indirizzo e che il provider sia disponibile."
                ) from error
            if dati.get("error"):
                raise RuntimeError(f"Errore del server Ollama: {dati['error']}")
            return dati

        dati = invia(corpo)
        messaggio = dati.get("message") or {}
        if not isinstance(messaggio, dict):
            raise RuntimeError("Risposta Ollama non valida: campo 'message' mancante.")
        contenuto = messaggio.get("content")
        if (
            not (isinstance(contenuto, str) and contenuto.strip())
            and messaggio.get("thinking")
            and "think" not in self.kwargs
        ):
            dati = invia({**corpo, "think": False})
            messaggio = dati.get("message") or {}
            if not isinstance(messaggio, dict):
                raise RuntimeError("Risposta Ollama non valida: campo 'message' mancante.")
        tool_calls = []
        for chiamata in messaggio.get("tool_calls") or []:
            funzione = chiamata.get("function") or {}
            nome = funzione.get("name")
            if not nome:
                continue
            argomenti = funzione.get("arguments") or {}
            if isinstance(argomenti, str):
                try:
                    argomenti = json.loads(argomenti)
                except ValueError:
                    argomenti = {}
            tool_calls.append(ToolCall(name=str(nome), arguments=dict(argomenti)))
        return LLMResponse(
            content=_ripulisci_testo_template(messaggio.get("content")),
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
            content=_ripulisci_testo_template(getattr(message, "content", None)),
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
