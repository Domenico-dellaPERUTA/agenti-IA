"""API pubblica della libreria agente.

Gli import qui raccolti permettono di costruire un agente con
``from AI import Agent, LMStudioProvider`` senza conoscere i moduli interni.
"""

from .core import Agent, LLMProvider, LLMResponse, ToolCall, ToolExecutionError
from .orchestration import (
    AgentOrchestrator,
    OrchestrationError,
    OrchestrationResult,
    TaskResult,
    TaskSpec,
)
from .providers import LMStudioProvider, OllamaProvider, OpenAIProvider
from .web import InternetAccess
from .application import AI

# ``__all__`` documenta i nomi supportati dall'importazione pubblica ``AI.*``.
__all__ = [
    "Agent",
    "AgentOrchestrator",
    "AI",
    "InternetAccess",
    "LLMProvider",
    "LLMResponse",
    "OrchestrationError",
    "OrchestrationResult",
    "TaskResult",
    "TaskSpec",
    "ToolExecutionError",
    "ToolCall",
    "OllamaProvider",
    "OpenAIProvider",
    "LMStudioProvider",
]
