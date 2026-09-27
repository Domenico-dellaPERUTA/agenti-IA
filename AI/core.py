from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from functools import wraps
import json
from collections.abc import Callable, Mapping
from pathlib import Path
import re
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
    coordina l'esecuzione delle chiamate generate dal modello. Se viene
    specificata una sandbox, registra anche le nove azioni sui file; le singole
    azioni possono essere sostituite passando callback in ``actions``.

    Esempio:
        >>> provider = MioProvider()
        >>> agent = Agent(provider, system_prompt="Sei utile.")
        >>> agent.send("Ciao!")
        >>> risposta = agent.run()
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str | None = None,
        *,
        sandbox: str | Path | None = None,
        actions: Mapping[str, Callable[..., Any]] | None = None,
    ):
        self.provider = provider
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.tools: dict[str, Callable] = {}
        self.sandbox = Path(sandbox).expanduser().resolve() if sandbox is not None else None

        if self.sandbox is not None and not self.sandbox.is_dir():
            raise ValueError(f"La cartella sandbox non esiste: {self.sandbox}")

        if actions and self.sandbox is None:
            raise ValueError(
                "Per configurare azioni personalizzate è necessario specificare la sandbox."
            )

        default_actions = {
            "list_files": self.list_files,
            "search_text": self.search_text,
            "read_file_excerpt": self.read_file_excerpt,
            "create_file": self.create_file,
            "append_to_file": self.append_to_file,
            "move_file": self.move_file,
            "extract_information": self.extract_information,
            "compare_files": self.compare_files,
            "prepare_tasks": self.prepare_tasks,
        }

        if actions:
            unknown_actions = set(actions) - set(default_actions)
            if unknown_actions:
                names = ", ".join(sorted(unknown_actions))
                raise ValueError(f"Azioni personalizzate non riconosciute: {names}")
            invalid_actions = [name for name, action in actions.items() if not callable(action)]
            if invalid_actions:
                names = ", ".join(sorted(invalid_actions))
                raise TypeError(f"Le azioni personalizzate devono essere funzioni: {names}")

        if self.sandbox is not None:
            for name, default_action in default_actions.items():
                action = (actions or {}).get(name, default_action)
                if action is not default_action:
                    callback = action
                    action = wraps(default_action)(
                        lambda _callback=callback, **kwargs: _callback(**kwargs)
                    )
                self.add_tool(action, name=name)

    def add_tool(self, fn: Callable, *, name: str | None = None):
        self.tools[name or fn.__name__] = fn

    def _sandbox_path(self, file_path: str) -> Path:
        if self.sandbox is None:
            raise RuntimeError("Questa azione richiede una cartella sandbox.")
        path = (self.sandbox / file_path).resolve()
        if not path.is_relative_to(self.sandbox):
            raise PermissionError("Accesso non autorizzato: il percorso è fuori dalla sandbox.")
        return path

    def list_files(self, directory: str = "") -> list[str]:
        """Elenca i file di una cartella della sandbox e delle sue sottocartelle."""
        folder = self._sandbox_path(directory)
        if not folder.is_dir():
            raise NotADirectoryError(f"La cartella '{directory}' non esiste nella sandbox.")
        sandbox = self.sandbox
        assert sandbox is not None
        return sorted(
            path.relative_to(sandbox).as_posix()
            for path in folder.rglob("*")
            if path.is_file() and path.resolve().is_relative_to(sandbox)
        )

    def search_text(self, query: str, file_extension: str = "") -> list[str]:
        """Cerca una frase nei file di testo della sandbox, opzionalmente filtrati per estensione."""
        if not query:
            raise ValueError("La ricerca non può essere vuota.")
        if file_extension and not file_extension.startswith("."):
            file_extension = f".{file_extension}"

        results = []
        for relative_path in self.list_files():
            path = self._sandbox_path(relative_path)
            if file_extension and path.suffix != file_extension:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"Il file '{relative_path}' non è un testo UTF-8 valido."
                ) from error
            results.extend(
                f"{relative_path}:{line_number}: {line}"
                for line_number, line in enumerate(lines, start=1)
                if query.casefold() in line.casefold()
            )
        return results

    def read_file_excerpt(
        self, file_path: str, start_line: int = 1, end_line: int = 50
    ) -> str:
        """Legge un intervallo di righe di un file della sandbox, numerate da 1."""
        if start_line < 1 or end_line < start_line:
            raise ValueError("Intervallo di righe non valido.")
        path = self._sandbox_path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Il file '{file_path}' non esiste nella sandbox.")
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(lines[start_line - 1:end_line], start=start_line)
        )

    def create_file(self, file_path: str, content: str) -> str:
        """Crea un nuovo file nella sandbox senza sovrascrivere file esistenti."""
        path = self._sandbox_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(content)
        return f"Creato: {path.relative_to(self._sandbox_root()).as_posix()}"

    def append_to_file(self, file_path: str, content: str) -> str:
        """Aggiunge testo alla fine di un file della sandbox, creandolo se manca."""
        path = self._sandbox_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(content)
        return f"Aggiornato: {path.relative_to(self._sandbox_root()).as_posix()}"

    def move_file(self, source_path: str, destination_path: str) -> str:
        """Sposta o rinomina un file della sandbox senza sovrascrivere la destinazione."""
        source = self._sandbox_path(source_path)
        destination = self._sandbox_path(destination_path)
        if not source.is_file():
            raise FileNotFoundError(f"Il file '{source_path}' non esiste nella sandbox.")
        if source == destination:
            raise ValueError("Origine e destinazione coincidono.")
        if destination.exists():
            raise FileExistsError(f"La destinazione '{destination_path}' esiste già.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        return f"Spostato in: {destination.relative_to(self._sandbox_root()).as_posix()}"

    def _sandbox_root(self) -> Path:
        if self.sandbox is None:
            raise RuntimeError("Questa azione richiede una cartella sandbox.")
        return self.sandbox

    def extract_information(self, file_path: str, query: str) -> list[dict[str, Any]]:
        """Estrae le righe che contengono il testo richiesto, con il numero di riga."""
        path = self._sandbox_path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Il file '{file_path}' non esiste nella sandbox.")
        if not query:
            raise ValueError("Il testo da cercare non può essere vuoto.")
        lines = path.read_text(encoding="utf-8").splitlines()
        return [
            {"line": line_number, "text": line}
            for line_number, line in enumerate(lines, start=1)
            if query.casefold() in line.casefold()
        ]

    def compare_files(self, first_path: str, second_path: str) -> str:
        """Confronta due file di testo della sandbox e restituisce le differenze."""
        first = self._sandbox_path(first_path)
        second = self._sandbox_path(second_path)
        if not first.is_file() or not second.is_file():
            raise FileNotFoundError("Entrambi i file da confrontare devono esistere nella sandbox.")
        first_lines = first.read_text(encoding="utf-8").splitlines()
        second_lines = second.read_text(encoding="utf-8").splitlines()
        diff = difflib.unified_diff(
            first_lines,
            second_lines,
            fromfile=first.relative_to(self._sandbox_root()).as_posix(),
            tofile=second.relative_to(self._sandbox_root()).as_posix(),
            lineterm="",
        )
        return "\n".join(diff) or "I file non presentano differenze."

    def prepare_tasks(self, file_path: str) -> list[str]:
        """Estrae da un file righe in formato checklist o con indicatori di attività."""
        path = self._sandbox_path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Il file '{file_path}' non esiste nella sandbox.")
        task_pattern = re.compile(
            r"^\s*(?:[-*+]\s+\[[ xX]\]\s+|\d+[.)]\s+|[-*+]\s+)", re.IGNORECASE
        )
        task_words = (
            "todo",
            "task",
            "da fare",
            "fare ",
            "chiamare",
            "inviare",
            "verificare",
            "completare",
            "preparare",
            "creare",
            "aggiornare",
            "contattare",
            "implementare",
        )
        tasks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            item = task_pattern.sub("", line).strip()
            if item and (
                task_pattern.match(line)
                or any(word in item.casefold() for word in task_words)
            ):
                tasks.append(item)
        return tasks

    def send(self, message: str, role: str = "user"):
        self.messages.append({"role": role, "content": message})

    def _stringify_tool_result(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)
        return json.dumps(value, ensure_ascii=False)

    def run(
        self,
        on_response: Callable[[LLMResponse], None] | None = None,
    ) -> str | None:
        while True:
            response = self.provider.complete(self.messages, list(self.tools.values()))
            if on_response is not None:
                on_response(response)

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
