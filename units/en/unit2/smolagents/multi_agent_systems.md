# Multi-Agent Systems

Multi-agent systems enable specialized agents to collaborate on complex tasks, improving modularity, scalability, and robustness. Instead of relying on a single agent for all operations, tasks are distributed among agents with distinct capabilities.

In **smolagents**, different agents can be combined to generate Python code, call external tools, perform web searches, and more. By orchestrating these agents, we can create powerful workflows, such as:

- A **Manager Agent** that delegates tasks.  
- A **Code Interpreter Agent** that executes Python code.  
- A **Web Search Agent** that retrieves information from the internet.  

The diagram below illustrates a simple multi-agent architecture where a **Manager Agent** coordinates a **Code Interpreter Tool** and a **Web Search Agent**, which in turn utilizes tools like `Web Search` and `Visit Webpage` to gather relevant information.

<img src="https://mermaid.ink/img/pako:eNp1kc1qhTAQRl9FUiQb8wIpdNO76eKubrmFks1oRg3VSYgjpYjv3lFL_2hnMWQOJwn5sqgmelRWleUSKLAtFs09jqhtoWuYUFfFAa6QA9QDTnpzamheuhxn8pt40-6l13UtS0ddhtQXj6dbR4XUGQg6zEYasTF393KjeSDGnDJKNxzj8I_7hLW5IOSmP9CH9hv_NL-d94d4DVNg84p1EnK4qlIj5hGClySWbadT-6OdsrL02MI8sFOOVkciw8zx8kaNspxnrJQE0fXKtjBMMs3JA-MpgOQwftIE9Bzj14w-cMznI_39E9Z3p0uFoA?type=png" style='background: white;'>

## Multi-Agent Systems in Action  

A multi-agent system consists of multiple specialized agents working together under the coordination of an **Orchestrator Agent**. This approach enables complex workflows by distributing tasks among agents with distinct roles.  

For example, a **Multi-Agent RAG system** can integrate:  
- A **Web Agent** for browsing the internet.  
- A **Retriever Agent** for fetching information from knowledge bases.  
- An **Image Generation Agent** for producing visuals.  

All of these agents operate under an orchestrator that manages task delegation and interaction.  

### Building a Multi-Agent System with `smolagents`  

To create a multi-agent system in `smolagents`, we start by defining individual `CodeAgent` instances, each responsible for a specific task. These agents are then managed by an **Orchestrator Agent**, which acts as the central coordinator.  

The orchestrator is initialized with a `managed_agents` attribute, listing the agents it controls. This modular approach allows for flexible and scalable multi-agent architectures.  

```python
# https://huggingface.co/learn/cookbook/multiagent_rag_system

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent, managed_retriever_agent, managed_image_generation_agent],
    additional_authorized_imports=["time", "datetime", "PIL"],
)

# Many possible prompts!
manager_agent.run("How many years ago was Stripe founded?")
result = manager_agent.run("Improve this prompt, then generate an image of it.", prompt="A rabbit wearing a space suit")
manager_agent.run("How can I push a model to the Hub?")
manager_agent.run("How do you combine multiple adapters in peft?")
```

The library handles system management internally, so no additional code is needed.  

To better understand the multi-agent structure, `smolagents` can generate diagrams that visually represent the system's architecture:

```python
manager_agent.visualize()
```

This command will produce a diagram similar to:

```python
CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
├── ✅ Authorized imports: ['time', 'numpy', 'pandas']
├── 🛠️ Tools:
│   ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│   ┃ Name         ┃ Description                                   ┃ Arguments                                    ┃
│   ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   │ final_answer │ Provides a final answer to the given problem. │ answer (`any`): The final answer to the      │
│   │              │                                               │ problem                                      │
│   └──────────────┴───────────────────────────────────────────────┴──────────────────────────────────────────────┘
└── 🤖 Managed agents:
    └── search | ToolCallingAgent | Qwen/Qwen2.5-Coder-32B-Instruct
        ├── 📝 Description: Runs web searches for you. Give it your query as an argument.
        └── 🛠️ Tools:
            ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Name          ┃ Description                              ┃ Arguments                                ┃
            ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ web_search    │ Performs a duckduckgo web search based   │ query (`string`): The search query to    │
            │               │ on your query (think a Google search)    │ perform.                                 │
            │               │ then returns the top search results.     │                                          │
            │ visit_webpage │ Visits a webpage at the given URL and    │ url (`string`): The URL of the webpage   │
            │               │ returns its content as a markdown        │ to visit.                                │
            │               │ string.                                  │                                          │
            │ final_answer  │ Provides a final answer to the given     │ answer (`any`): The final answer to the  │
            │               │ problem.                                 │ problem                                  │
            └───────────────┴──────────────────────────────────────────┴──────────────────────────────────────────┘
```


## Further Reading  

- [Multi-Agent Systems](https://huggingface.co/docs/smolagents/main/en/examples/multiagents) – Overview of multi-agent systems.  
- [What is Agentic RAG?](https://weaviate.io/blog/what-is-agentic-rag) – Introduction to Agentic RAG.  
- [Multi-Agent RAG System 🤖🤝🤖 Recipe](https://huggingface.co/learn/cookbook/multiagent_rag_system) – Step-by-step guide to building a multi-agent RAG system.  
