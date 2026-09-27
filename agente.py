import os
from pathlib import Path

from AI import Agent, LMStudioProvider


PROMPT_INIZIALE = "Elenca i file disponibili nella sandbox e dimmi cosa contiene note.txt."


def crea_agente(cartella_sandbox: str | os.PathLike[str] | None = None) -> Agent:
    """Crea l'agente con accesso ai soli file della cartella selezionata."""
    percorso_sandbox = (
        cartella_sandbox if cartella_sandbox is not None else Path.cwd() / "sandbox"
    )
    if not str(percorso_sandbox).strip():
        raise ValueError("La cartella sandbox non può essere vuota.")
    sandbox = Path(percorso_sandbox).expanduser().resolve()
    if not sandbox.is_dir():
        raise ValueError(f"La cartella sandbox non esiste: {sandbox}")

    provider = LMStudioProvider(model="qwen3-4b-2507")  # OllamaProvider(model="qwen3:4b")
    return Agent(
        provider,
        system_prompt=(
            "Sei un assistente utile. Usa gli strumenti disponibili per lavorare "
            "solo sui file della sandbox. Non dichiarare di aver modificato file "
            "se lo strumento non ha confermato l'operazione. Modifica o sposta "
            "file solo se l'utente lo ha richiesto esplicitamente."
        ),
        sandbox=sandbox,
    )


def main() -> None:
    agent = crea_agente()
    agent.send(PROMPT_INIZIALE)
    print(agent.run())


if __name__ == "__main__":
    main()
