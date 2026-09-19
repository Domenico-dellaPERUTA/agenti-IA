import os

from AI import Agent, OllamaProvider, LMStudioProvider


def leggi_file(nome_file: str) -> str:
    """Legge un file solo se è dentro la cartella sandbox."""
    cartella_sandbox = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))
    percorso_file = os.path.abspath(os.path.join(cartella_sandbox, nome_file))

    if not (percorso_file.startswith(cartella_sandbox + os.sep) or percorso_file == cartella_sandbox):
        return "Errore: accesso non autorizzato. Il file richiesto è fuori dalla cartella sandbox."

    if not os.path.exists(percorso_file):
        return f"Errore: il file '{nome_file}' non esiste nella cartella sandbox."

    if not os.path.isfile(percorso_file):
        return f"Errore: '{nome_file}' non è un file valido."

    try:
        with open(percorso_file, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Errore durante la lettura del file: {e}"


provider = LMStudioProvider(model="qwen3-4b-2507") #OllamaProvider(model="qwen3:4b")
agent = Agent(provider, system_prompt="Sei un assistente utile. Usa gli strumenti quando serve.")
agent.add_tool(leggi_file)
agent.send("Leggi il file note.txt e mostrami solo il cosa contiene.")
print(agent.run())

