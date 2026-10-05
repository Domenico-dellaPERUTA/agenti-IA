"""Nucleo indipendente dal provider per conversazioni, strumenti e sandbox.

Un ``Agent`` delega il completamento dei messaggi a un ``LLMProvider``,
esegue le funzioni richieste dal modello e aggiunge i risultati alla
conversazione. Le interfacce qui definite sono il punto di estensione della
libreria per nuovi provider e nuovi strumenti.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import codecs
import difflib
from functools import wraps
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any


@dataclass
class ToolCall:
    """Rappresenta una chiamata a una funzione generata dal modello.

    Gli strumenti di un modello possono restituire più azioni da eseguire,
    ciascuna identificata dal nome e dai parametri da passare. Il provider
    converte la rappresentazione specifica del servizio in questa forma comune.

    Attributi:
        name: Nome registrato dello strumento che l'agente deve invocare.
        arguments: Argomenti nominati da passare alla funzione dello strumento.

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
    lista di tool call da eseguire in modo automatico. Un provider può
    restituire entrambe le parti se il modello produce testo insieme alle
    chiamate.

    Attributi:
        content: Testo finale o intermedio prodotto dal modello, se presente.
        tool_calls: Azioni richieste dal modello nel formato ``ToolCall``.

    Esempio:
        >>> risposta = LLMResponse(content="Ciao!")
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider:
    """Interfaccia astratta per i provider di modelli linguistici.

    Ogni provider specifico deve implementare il metodo ``complete`` per
    inviare messaggi al modello e trasformare la risposta in un
    ``LLMResponse``. I provider concreti mantengono così separati il protocollo
    di rete e il ciclo di orchestrazione dell'agente.

    Esempio:
        >>> class MioProvider(LLMProvider):
        ...     def complete(self, messages, tools=None):
        ...         return LLMResponse(content="Risposta locale")
        >>> provider = MioProvider()
    """

    def complete(self, messages: list[dict], tools: list[Callable] | None = None) -> LLMResponse:
        """Invia la conversazione al modello e normalizza la risposta.

        Args:
            messages: Cronologia in formato comune ``role``/``content``.
            tools: Funzioni disponibili da descrivere al modello, se supportate.

        Returns:
            Risposta normalizzata, contenente testo, chiamate a tool o entrambi.

        Raises:
            NotImplementedError: La classe base non implementa un provider.
        """
        raise NotImplementedError("Il provider deve implementare complete().")


class ToolExecutionError(RuntimeError):
    """Errore di uno strumento che l'agente può riportare al modello."""


class Agent:
    """Gestisce la conversazione e l'esecuzione di tool del modello.

    L'agent mantiene lo stato della chat, registra gli strumenti disponibili e
    coordina l'esecuzione delle chiamate generate dal modello. Se viene
    specificata una sandbox, registra anche le nove azioni sui file; le singole
    azioni possono essere sostituite passando callback in ``actions``.
    Gli strumenti per eseguire script vengono aggiunti solo se il chiamante
    fornisce una callback esplicita di approvazione. I provider di fonti
    raccolgono riferimenti da aggiungere alla risposta conclusiva.

    Args:
        provider: Implementazione del protocollo LLM da usare.
        system_prompt: Istruzioni iniziali opzionali inviate al modello.
        sandbox: Radice dei file accessibili agli strumenti di file.
        actions: Sostituzioni delle azioni standard, indicizzate per nome.
        allowed_tools: Nomi degli strumenti da registrare, se specificati.
        script_approval: Callback che mostra il codice e restituisce il consenso.
        script_output: Callback che riceve in tempo reale l'output dello script.

    Nota:
        La sandbox limita gli strumenti Python di file, ma non confina processi
        Bash: uno script approvato opera con i permessi dell'utente del processo.

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
        script_approval: Callable[[str, str], bool] | None = None,
        script_output: Callable[[str], None] | None = None,
        allowed_tools: Iterable[str] | None = None,
    ):
        """Inizializza stato, strumenti e limiti della sandbox dell'agente."""
        self.provider = provider
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.tools: dict[str, Callable] = {}
        self.source_providers: list[Callable[[], list[tuple[str, str]]]] = []
        # Risolve "~" e percorsi relativi subito, così tutti i tool usano una
        # radice assoluta stabile per la durata dell'agente.
        self.sandbox = Path(sandbox).expanduser().resolve() if sandbox is not None else None
        self.script_approval = script_approval
        self.script_output = script_output
        self._allowed_tools: frozenset[str] | None = None
        if allowed_tools is not None:
            self.restrict_tools(allowed_tools)

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
                    # Mantiene la firma/docstring del tool standard affinché
                    # il provider possa generare lo schema, delegando l'azione
                    # effettiva alla callback personalizzata.
                    action = wraps(default_action)(
                        lambda _callback=callback, **kwargs: _callback(**kwargs)
                    )
                self.add_tool(action, name=name)
            if self.script_approval is not None:
                self.add_tool(self.run_bash_script)

    def add_tool(self, fn: Callable, *, name: str | None = None):
        """Registra una funzione invocabile dal modello con il nome indicato.

        Se ``name`` è omesso si usa ``fn.__name__``. Registrare un nome già
        esistente sostituisce lo strumento precedente con quello nuovo.
        """
        tool_name = name or fn.__name__
        if self._allowed_tools is None or tool_name in self._allowed_tools:
            self.tools[tool_name] = fn

    def restrict_tools(self, allowed_tools: Iterable[str]) -> None:
        """Limita gli strumenti registrati e quelli che potranno essere aggiunti.

        Le restrizioni successive possono soltanto ridurre l'insieme consentito.
        """
        if isinstance(allowed_tools, (str, bytes)):
            raise TypeError("La allowlist degli strumenti deve essere una raccolta di nomi.")
        names = frozenset(allowed_tools)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("I nomi degli strumenti consentiti devono essere stringhe non vuote.")
        if self._allowed_tools is not None:
            names &= self._allowed_tools
        self._allowed_tools = names
        self.tools = {
            name: tool for name, tool in self.tools.items() if name in names
        }

    def add_source_provider(
        self, provider: Callable[[], list[tuple[str, str]]]
    ) -> None:
        """Registra una funzione che restituisce coppie ``(titolo, URL)``.

        Gli URL restituiti vengono deduplicati e allegati alla risposta finale
        quando almeno un provider produce fonti.
        """
        if not callable(provider):
            raise TypeError("Il provider delle fonti deve essere una funzione.")
        self.source_providers.append(provider)

    def _append_sources(self, content: str | None) -> str | None:
        """Aggiunge alla risposta le fonti aggregate senza ripetere gli URL."""
        sources: dict[str, str] = {}
        for provider in self.source_providers:
            for title, url in provider():
                sources.setdefault(url, title)
        if not sources:
            return content

        citations = "\n\nFonti web:\n" + "\n".join(
            f"- {title}: {url}" for url, title in sources.items()
        )
        return f"{content or ''}{citations}"

    def _sandbox_path(self, file_path: str) -> Path:
        """Risolvi un percorso relativo e rifiuta fughe fuori dalla sandbox.

        ``Path.resolve`` normalizza anche i componenti ``..`` e i collegamenti
        simbolici; il controllo successivo verifica la destinazione effettiva.
        """
        if self.sandbox is None:
            raise RuntimeError("Questa azione richiede una cartella sandbox.")
        path = (self.sandbox / file_path).resolve()
        if not path.is_relative_to(self.sandbox):
            raise PermissionError("Accesso non autorizzato: il percorso è fuori dalla sandbox.")
        return path

    def list_files(self, directory: str = "") -> list[str]:
        """Elenca percorsi relativi dei file interni alla cartella richiesta.

        I collegamenti simbolici che puntano fuori dalla sandbox vengono
        esclusi. Solleva ``NotADirectoryError`` se la cartella non esiste.
        """
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
        """Cerca una frase senza distinzione tra maiuscole e minuscole.

        Args:
            query: Testo non vuoto da individuare nelle righe.
            file_extension: Estensione facoltativa, con o senza punto iniziale.

        Returns:
            Righe corrispondenti nel formato ``percorso:numero: contenuto``.

        I file non UTF-8 causano un errore esplicito anziché essere ignorati.
        """
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
        """Legge un intervallo inclusivo di righe, con numerazione da 1.

        L'intervallo deve iniziare da una riga positiva e il termine non può
        precedere l'inizio; il percorso deve designare un file nella sandbox.
        """
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
        """Crea un nuovo file UTF-8 senza sovrascriverne uno esistente.

        Crea anche le cartelle intermedie. L'apertura in modalità esclusiva
        rende atomica la protezione contro la sovrascrittura concorrente.
        """
        path = self._sandbox_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(content)
        return f"Creato: {path.relative_to(self._sandbox_root()).as_posix()}"

    def append_to_file(self, file_path: str, content: str) -> str:
        """Aggiunge testo UTF-8 alla fine di un file, creandolo se necessario.

        Le cartelle intermedie sono create automaticamente; il percorso resta
        vincolato alla sandbox come per gli altri strumenti di file.
        """
        path = self._sandbox_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(content)
        return f"Aggiornato: {path.relative_to(self._sandbox_root()).as_posix()}"

    def move_file(self, source_path: str, destination_path: str) -> str:
        """Sposta o rinomina un file senza sovrascrivere la destinazione.

        Rifiuta file sorgente assenti, destinazioni già occupate e operazioni
        che indicano lo stesso percorso per origine e destinazione.
        """
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

    def run_bash_script(self, script: str) -> str:
        """Esegue codice Bash dopo approvazione, usando una cartella temporanea.

        Il modello passa direttamente il codice, senza creare un file persistente.
        Il contenuto è limitato a 64 KiB, salvato in un file temporaneo e
        mostrato integralmente alla callback prima dell'esecuzione. Il comando
        usa la sandbox come directory di lavoro, ma la sandbox non isola i
        permessi del processo. Il gruppo viene terminato dopo 120 secondi;
        l'output viene passato progressivamente a ``script_output`` e limitato
        a 1 MiB. La cartella temporanea viene eliminata all'uscita dal blocco,
        sia in caso di approvazione, rifiuto o eccezione. Per esempio, una
        richiesta esplicita di scansione locale con nmap può passare il comando
        limitato a ``127.0.0.1`` o ``::1`` da mostrare all'utente.

        Args:
            script: Codice Bash completo che l'utente ha chiesto di eseguire.

        Returns:
            Esito con codice di uscita oppure indicazione di annullamento.

        Raises:
            PermissionError: Nessuna callback di approvazione o processo root.
            ValueError: Script troppo grande o contenente byte NUL.
            RuntimeError: Sistema operativo non POSIX o output oltre il limite.
            TimeoutError: Lo script supera il limite temporale.
        """
        if self.script_approval is None:
            raise PermissionError("L'esecuzione degli script non è abilitata senza approvazione GUI.")
        if not isinstance(script, str):
            raise TypeError("Il contenuto dello script deve essere testo.")
        script_bytes = script.encode("utf-8")
        if len(script_bytes) > 64 * 1024:
            raise ValueError("Lo script supera il limite consentito di 64 KiB.")
        if "\x00" in script:
            raise ValueError("Lo script contiene caratteri NUL non validi.")

        if os.name != "posix":
            raise RuntimeError("L'esecuzione di script Bash è supportata solo su macOS e Linux.")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            raise PermissionError("Per sicurezza, gli script non vengono eseguiti come root.")

        with tempfile.TemporaryDirectory(prefix="agente-script-") as temporary_directory:
            # Il contesto temporaneo garantisce il tentativo di pulizia anche
            # se l'approvazione rifiuta o l'esecuzione termina con eccezione.
            script_path = Path(temporary_directory) / "script.sh"
            script_path.write_bytes(script_bytes)
            nome_script = "script temporaneo (script.sh)"
            if not self.script_approval(nome_script, script):
                return f"Esecuzione annullata dall'utente: {nome_script}"
            return self._execute_bash_script(script_path, nome_script)

    def _execute_bash_script(self, script_path: Path, display_name: str) -> str:
        """Avvia Bash e inoltra l'output senza attendere che il buffer si riempia.

        La lettura non bloccante tramite ``select`` permette di controllare
        durata e byte totali mentre lo script è in esecuzione. Il gruppo di
        processi viene terminato in caso di timeout, output eccessivo o errore
        della callback di log.
        """
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(self._sandbox_root()),
            "TMPDIR": str(script_path.parent),
            "LANG": "C",
        }
        process = subprocess.Popen(
            ["bash", "--noprofile", "--norc", str(script_path)],
            cwd=self._sandbox_root(),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        assert process.stdout is not None
        started_at = time.monotonic()
        timed_out = False
        output_limit_exceeded = False
        output_size = 0
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def terminate_process_group() -> None:
            """Invia prima SIGTERM e poi SIGKILL al gruppo, aspettando la chiusura."""
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                if process.poll() is None:
                    process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if process.poll() is None:
                    process.kill()
            process.wait()

        try:
            while True:
                if time.monotonic() - started_at > 120:
                    timed_out = True
                    terminate_process_group()
                    break

                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if not ready:
                    continue
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break

                output_size += len(chunk)
                if output_size > 1024 * 1024:
                    output_limit_exceeded = True
                    terminate_process_group()
                    break
                if self.script_output is not None:
                    decoded = decoder.decode(chunk)
                    if decoded:
                        self.script_output(decoded)
        except BaseException:
            terminate_process_group()
            raise
        finally:
            process.stdout.close()

        return_code = process.wait()
        if timed_out:
            raise TimeoutError(
                "Script terminato dopo aver superato il limite di 120 secondi: "
                f"{display_name}"
            )
        if output_limit_exceeded:
            raise RuntimeError(
                "Script terminato dopo aver superato il limite di output di 1 MiB: "
                f"{display_name}"
            )
        remaining_output = decoder.decode(b"", final=True)
        if remaining_output and self.script_output is not None:
            self.script_output(remaining_output)
        return f"Script temporaneo terminato con codice {return_code}."

    def _sandbox_root(self) -> Path:
        """Restituisce la radice validata o segnala che manca la sandbox."""
        if self.sandbox is None:
            raise RuntimeError("Questa azione richiede una cartella sandbox.")
        return self.sandbox

    def extract_information(self, file_path: str, query: str) -> list[dict[str, Any]]:
        """Estrae tutte le righe che contengono la query, senza distinzione di caso.

        Ogni risultato è un dizionario con numero di riga (base 1) e testo
        originale, così i chiamanti possono elaborare entrambi separatamente.
        """
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
        """Confronta due file UTF-8 e restituisce un diff unificato.

        Se i contenuti coincidono restituisce un messaggio esplicito invece
        di una stringa vuota.
        """
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
        """Estrae elementi di checklist e righe che sembrano attività.

        Riconosce marcatori Markdown, elenchi numerati e parole chiave italiane
        o inglesi; restituisce il testo ripulito senza il marcatore iniziale.
        """
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
        """Aggiunge un messaggio alla cronologia senza contattare il provider."""
        self.messages.append({"role": role, "content": message})

    def _stringify_tool_result(self, value):
        """Converte i risultati dei tool in contenuto testuale per la chat."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)
        return json.dumps(value, ensure_ascii=False)

    def run(
        self,
        on_response: Callable[[LLMResponse], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
    ) -> str | None:
        """Interroga il modello e gestisce le chiamate agli strumenti finché termina.

        Per ogni iterazione invia cronologia e descrizioni dei tool al provider.
        Le tool call vengono eseguite nell'ordine ricevuto; i risultati entrano
        nella cronologia con ruolo ``tool`` e vengono inviati al modello al
        passaggio seguente. Gli errori diversi da una collisione di file
        propagano al chiamante, così la GUI o la CLI possono mostrarli.

        Args:
            on_response: Callback facoltativa invocata per ogni risposta del modello.
            on_tool_result: Callback facoltativa ``(nome, risultato)`` dopo ogni tool.

        Returns:
            Testo conclusivo del modello con eventuali fonti, oppure ``None``.
        """
        while True:
            response = self.provider.complete(self.messages, list(self.tools.values()))
            if on_response is not None:
                on_response(response)

            if response.content is not None and response.content != "":
                self.messages.append({"role": "assistant", "content": response.content})

            if not response.tool_calls:
                final_content = self._append_sources(response.content)
                if final_content != response.content and final_content is not None:
                    if (
                        self.messages
                        and self.messages[-1].get("role") == "assistant"
                        and self.messages[-1].get("content") == response.content
                    ):
                        self.messages[-1]["content"] = final_content
                    else:
                        self.messages.append(
                            {"role": "assistant", "content": final_content}
                        )
                return final_content

            for call in response.tool_calls:
                tool_fn = self.tools.get(call.name)
                if tool_fn is None:
                    raise ValueError(f"Tool non registrata: {call.name}")

                try:
                    risultato = self._stringify_tool_result(
                        tool_fn(**call.arguments)
                    )
                except FileExistsError as error:
                    risultato = (
                        f"Errore: {error}. Il file esiste già e non è stato "
                        "sovrascritto. Scegli un percorso diverso e riprova."
                    )
                except ToolExecutionError as error:
                    risultato = f"Errore dello strumento: {error}"

                if on_tool_result is not None:
                    on_tool_result(call.name, risultato)
                self.messages.append({
                    "role": "tool",
                    "tool_name": call.name,
                    "content": risultato,
                })
