"""Supporto condiviso per la configurazione del provider LLM.

GUI e CLI offrono una selezione guidata del provider; la logica di creazione,
validazione ed elenco dei modelli vive qui e non nei frontend, così le regole
(quali campi servono, quali sono i predefiniti, come si legge l'elenco dei
modelli del server) restano in un unico posto.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .core import LLMProvider
from .providers import LMStudioProvider, OllamaProvider, OpenAIProvider

# Una scheda per provider: chiave stabile, titolo leggibile e valori dei campi
# preimpostati ai valori tipici. La chiave API non ha mai un predefinito.
SCHEDE_PROVIDER: dict[str, tuple[str, dict[str, str]]] = {
    "lmstudio": (
        "LM Studio",
        {
            "base_url": "http://localhost:1234/v1",
            "modello": "ornith-1.5-9b-uncensored",
        },
    ),
    "ollama": (
        "Ollama",
        {"modello": "qwen3:4b", "base_url": "http://localhost:11434"},
    ),
    "openai": (
        "OpenAI API",
        {"modello": "gpt-4o-mini", "api_key": "", "base_url": ""},
    ),
}

MODELLI_PREDEFINITI_OPENAI = ["gpt-4o-mini", "gpt-4o"]


def crea_provider(scheda: str, config: dict[str, str]) -> LLMProvider:
    """Crea il provider della scheda richiesta, validando i campi.

    Solleva ``ValueError`` con un messaggio comprensibile quando manca il
    modello o la chiave API. Il provider Ollama funziona anche senza il
    pacchetto Python ``ollama`` (ripiega sulla API HTTP del demone).
    """
    modello = config.get("modello", "").strip()
    if not modello:
        raise ValueError("Inserisci il nome del modello nella scheda corrente.")
    if scheda == "lmstudio":
        base_url = config.get("base_url", "").strip() or "http://localhost:1234/v1"
        return LMStudioProvider(model=modello, base_url=base_url)
    if scheda == "ollama":
        host = config.get("base_url", "").strip() or None
        return OllamaProvider(model=modello, host=host)
    if scheda == "openai":
        chiave = (
            config.get("api_key", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        if not chiave:
            raise ValueError(
                "Chiave OpenAI mancante: inseriscila nel campo o imposta "
                "la variabile d'ambiente OPENAI_API_KEY."
            )
        base_url = config.get("base_url", "").strip() or None
        return OpenAIProvider(model=modello, api_key=chiave, base_url=base_url)
    raise ValueError(f"Provider sconosciuto: {scheda}")


def modelli_ollama(host: str) -> list[str]:
    """Elenca i modelli del demone Ollama via API HTTP, senza l'SDK.

    ``GET /api/tags`` equivale al comando ``ollama list`` e basta la libreria
    standard, così la verifica funziona anche quando il pacchetto Python
    ``ollama`` (facoltativo) non è installato.
    """
    host = (host or "").strip().rstrip("/") or "http://localhost:11434"
    richiesta = urllib.request.Request(
        f"{host}/api/tags", headers={"User-Agent": "agenti-IA"}
    )
    try:
        with urllib.request.urlopen(richiesta, timeout=5) as risposta:
            dati = json.loads(risposta.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        motivo = getattr(error, "reason", error)
        raise ValueError(
            f"Server Ollama non raggiungibile su {host} ({motivo}). "
            "Avvia Ollama o correggi l'indirizzo del server."
        ) from error
    return [
        str(modello["name"])
        for modello in dati.get("models", [])
        if isinstance(modello, dict) and modello.get("name")
    ]


def elenca_modelli(scheda: str, config: dict[str, str]) -> list[str]:
    """Interroga l'elenco modelli del provider (operazione di rete)."""
    if scheda == "ollama":
        return modelli_ollama(config.get("base_url", ""))
    provider = crea_provider(scheda, config)
    client = getattr(provider, "client", None)
    if client is None:
        raise ValueError(f"Il provider {scheda} non espone un elenco modelli.")
    elenco = client.models.list()
    return [str(m.id) for m in getattr(elenco, "data", []) if getattr(m, "id", None)]


def descrivi_provider(scheda: str, config: dict[str, str]) -> str:
    """Riassunto testuale del provider selezionato, per il log."""
    titolo = SCHEDE_PROVIDER[scheda][0]
    modello = config.get("modello", "").strip()
    base_url = config.get("base_url", "").strip()
    if scheda == "lmstudio":
        return f"{titolo} · modello {modello} @ {base_url or 'http://localhost:1234/v1'}"
    if scheda == "ollama":
        return f"{titolo} · modello {modello} @ {base_url or 'http://localhost:11434'}"
    if scheda == "openai" and base_url:
        return f"{titolo} · modello {modello} @ {base_url}"
    return f"{titolo} · modello {modello}"
