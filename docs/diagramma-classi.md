# Diagramma UML delle classi

Il diagramma mostra le classi principali del package `AI` e le relazioni
utilizzate durante la creazione e l'orchestrazione degli agenti.

```mermaid
classDiagram
    class AI {
        +LLMProvider provider
        +Path sandbox
        +str initial_prompt
        +str base_system_prompt
        +create_agent(script_approval, script_output, system_prompt, allowed_tools) Agent
        +create_orchestrator(script_approval, script_output, max_workers, max_tasks) AgentOrchestrator
    }

    class Agent {
        +LLMProvider provider
        +list~dict~ messages
        +dict~str, Callable~ tools
        +Path sandbox
        +send(message)
        +run() str
        +add_tool(fn)
        +add_source_provider(provider)
    }

    class LLMProvider {
        <<protocol>>
        +complete(messages, tools) LLMResponse
    }

    class OllamaProvider {
        +str model
        +complete(messages, tools) LLMResponse
    }

    class OpenAIProvider {
        +str model
        +complete(messages, tools) LLMResponse
    }

    class LMStudioProvider {
        +str model
        +str base_url
    }

    class AgentOrchestrator {
        +Callable agent_factory
        +int max_workers
        +int max_tasks
        +run(request, on_event) OrchestrationResult
    }

    class InternetAccess {
        +web_search(query) str
        +read_webpage(url) str
        +sources() list~tuple~
    }

    class LLMResponse {
        +str content
        +list~ToolCall~ tool_calls
    }

    class ToolCall {
        +str name
        +dict arguments
    }

    class TaskSpec {
        +str id
        +str title
        +str instructions
        +str context
    }

    class TaskResult {
        +str task_id
        +str status
        +str output
        +str error
    }

    class OrchestrationResult {
        +str answer
        +list~TaskResult~ tasks
        +str mode
    }

    LLMProvider <|.. OllamaProvider
    LLMProvider <|.. OpenAIProvider
    OpenAIProvider <|-- LMStudioProvider
    AI --> LLMProvider : riceve provider configurato
    AI ..> Agent : crea istanze
    AI ..> InternetAccess : registra strumenti e fonti
    AI ..> AgentOrchestrator : crea
    Agent --> LLMProvider : usa
    AgentOrchestrator ..> Agent : factory crea agenti indipendenti
    AgentOrchestrator ..> TaskSpec : pianifica
    AgentOrchestrator ..> TaskResult : raccoglie
    AgentOrchestrator ..> OrchestrationResult : restituisce
    LLMProvider ..> LLMResponse : restituisce
    LLMResponse *-- ToolCall : contiene
```

`AI` condivide la stessa istanza di provider tra gli agenti creati per una
richiesta; ogni `Agent` mantiene invece messaggi, strumenti e fonti per il
proprio contesto. Di conseguenza, un provider passato a `AI` deve poter
gestire chiamate concorrenti quando l'orchestratore esegue più worker in
parallelo.
