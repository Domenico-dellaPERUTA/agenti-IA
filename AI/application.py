"""Composizione dei componenti AI per le applicazioni dimostrative."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import os
from pathlib import Path

from .core import Agent, LLMProvider
from .orchestration import AgentOrchestrator
from .web import InternetAccess

_SCRIPT_SYSTEM_PROMPT = (
    " Gli script Bash possono essere eseguiti solo dopo che l'utente "
    "ha approvato esplicitamente il codice mostrato dalla GUI o nel "
    "terminale; non aggirare né anticipare tale conferma. Quando la "
    "richiesta dell'utente richiede esplicitamente uno script, invoca "
    "run_bash_script passando direttamente il codice nello strumento: "
    "il tool crea un file temporaneo, mostra lo script per "
    "l'approvazione e poi elimina il temporaneo al termine. Non creare "
    "prima un file .sh con create_file. Non "
    "rispondere che non puoi avviare programmi esterni o strumenti "
    "di rete se è disponibile questo tool. Per una richiesta di "
    "scansione nmap locale, puoi preparare un comando nmap circoscritto "
    "a 127.0.0.1 (o ::1); per un indirizzo o una rete LAN diversi, "
    "chiedi prima all'utente di specificare il target e assicurarsi "
    "di averne l'autorizzazione. Non scansionare target pubblici o "
    "più ampi di quelli richiesti. Riporta gli output e gli errori "
    "effettivi del tool. Se create_file segnala che il file esiste "
    "già per un normale file richiesto dall'utente, non ripetere la "
    "stessa chiamata: scegli un nome di file diverso e prosegui."
)


class AI:
    """Factory configurabile che crea agenti e orchestratori per un'app."""

    def __init__(
        self,
        provider: LLMProvider,
        cartella_sandbox: str | os.PathLike[str] | None = None,
        *,
        initial_prompt: str,
        base_system_prompt: str,
    ) -> None:
        percorso_sandbox = (
            cartella_sandbox
            if cartella_sandbox is not None
            else Path.cwd() / "sandbox"
        )
        if not str(percorso_sandbox).strip():
            raise ValueError("La cartella sandbox non può essere vuota.")
        sandbox = Path(percorso_sandbox).expanduser().resolve()
        if not sandbox.is_dir():
            raise ValueError(f"La cartella sandbox non esiste: {sandbox}")
        if not isinstance(provider, LLMProvider):
            raise TypeError("provider deve essere un'istanza di LLMProvider.")
        if not isinstance(initial_prompt, str):
            raise TypeError("Il prompt iniziale deve essere una stringa.")
        if not isinstance(base_system_prompt, str):
            raise TypeError("Il prompt di sistema deve essere una stringa.")

        self.sandbox = sandbox
        self.provider = provider
        self.initial_prompt = initial_prompt
        self.base_system_prompt = base_system_prompt

    def create_agent(
        self,
        *,
        script_approval: Callable[[str, str], bool] | None = None,
        script_output: Callable[[str], None] | None = None,
        system_prompt: str | None = None,
        allowed_tools: Iterable[str] | None = None,
    ) -> Agent:
        """Crea un agente usando le impostazioni condivise dell'applicazione."""
        prompt_sistema = self.base_system_prompt
        if script_approval is not None:
            prompt_sistema += _SCRIPT_SYSTEM_PROMPT
        else:
            prompt_sistema += (
                " In questa modalità non è disponibile uno strumento di esecuzione "
                "script: non affermare di aver eseguito comandi."
            )
        if system_prompt:
            prompt_sistema += "\n\n" + system_prompt

        agent = Agent(
            self.provider,
            system_prompt=prompt_sistema,
            sandbox=self.sandbox,
            script_approval=script_approval,
            script_output=script_output,
            allowed_tools=allowed_tools,
        )
        internet = InternetAccess()
        agent.add_tool(internet.web_search)
        agent.add_tool(internet.read_webpage)
        agent.add_source_provider(internet.sources)
        return agent

    def create_orchestrator(
        self,
        *,
        script_approval: Callable[[str, str], bool] | None = None,
        script_output: Callable[[str], None] | None = None,
        max_workers: int = 2,
        max_tasks: int = 5,
    ) -> AgentOrchestrator:
        """Crea l'orchestratore con una factory di agenti indipendenti."""
        return AgentOrchestrator(
            agent_factory=lambda system_prompt: self.create_agent(
                script_approval=script_approval,
                script_output=script_output,
                system_prompt=system_prompt,
            ),
            max_workers=max_workers,
            max_tasks=max_tasks,
        )
