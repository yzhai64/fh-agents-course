# Multi-Agent Systems

Multi-agent systems enable specialized agents to collaborate on complex tasks, improving modularity, scalability, and robustness. Instead of relying on a single agent for all operations, tasks are distributed among agents with distinct capabilities.

In **smolagents**, different agents can be combined to generate Python code, call external tools, perform web searches, and more. By orchestrating these agents, we can create powerful workflows, such as:

- A **Manager Agent** that delegates tasks.  
- A **Code Interpreter Agent** that executes Python code.  
- A **Web Search Agent** that retrieves information from the internet.  

The diagram below illustrates a simple multi-agent architecture where a **Manager Agent** coordinates a **Code Interpreter Tool** and a **Web Search Agent**, which in turn utilizes tools like `Web Search` and `Visit Webpage` to gather relevant information.

```bash
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
Code Interpreter            +------------------+
    tool                    | Web Search agent |
                            +------------------+
                               |            |
                        Web Search tool     |
                                   Visit webpage tool
```

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

## Further Reading  

- [Multi-Agent Systems](https://huggingface.co/docs/smolagents/main/en/examples/multiagents) – Overview of multi-agent systems.  
- [What is Agentic RAG?](https://weaviate.io/blog/what-is-agentic-rag) – Introduction to Agentic RAG.  
- [Multi-Agent RAG System 🤖🤝🤖 Recipe](https://huggingface.co/learn/cookbook/multiagent_rag_system) – Step-by-step guide to building a multi-agent RAG system.  
