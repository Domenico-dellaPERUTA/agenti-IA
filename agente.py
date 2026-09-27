import os
from pathlib import Path

from AI import Agent, OllamaProvider, LMStudioProvider


PROMPT_INIZIALE = "Leggi il file note.txt e mostrami solo il cosa contiene."


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

    def leggi_file(nome_file: str) -> str:
        """Legge un file solo se è dentro la cartella sandbox."""
        percorso_file = (sandbox / nome_file).resolve()

        if not percorso_file.is_relative_to(sandbox):
            return "Errore: accesso non autorizzato. Il file richiesto è fuori dalla cartella sandbox."

        if not percorso_file.exists():
            return f"Errore: il file '{nome_file}' non esiste nella cartella sandbox."

        if not percorso_file.is_file():
            return f"Errore: '{nome_file}' non è un file valido."

        try:
            return percorso_file.read_text(encoding="utf-8")
        except OSError as error:
            return f"Errore durante la lettura del file: {error}"

    provider = LMStudioProvider(model="qwen3-4b-2507")  # OllamaProvider(model="qwen3:4b")
    agent = Agent(provider, system_prompt="Sei un assistente utile. Usa gli strumenti quando serve.")
    agent.add_tool(leggi_file)
    return agent


def main() -> None:
    agent = crea_agente()
    agent.send(PROMPT_INIZIALE)
    print(agent.run())


if __name__ == "__main__":
    main()
