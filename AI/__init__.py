"""API pubblica della libreria agente.

Gli import qui raccolti permettono di costruire un agente con
``from AI import Agent, LMStudioProvider`` senza conoscere i moduli interni.
"""

from .core import Agent, LLMProvider, LLMResponse, ToolCall
from .providers import LMStudioProvider, OllamaProvider, OpenAIProvider
from .web import InternetAccess

# ``__all__`` documenta i nomi supportati dall'importazione pubblica ``AI.*``.
__all__ = [
    "Agent",
    "InternetAccess",
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "OllamaProvider",
    "OpenAIProvider",
    "LMStudioProvider",
]
