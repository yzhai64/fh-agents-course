# Code Agents

Code agents are the default agent type in smolagents. They generate Python tool calls to perform actions, improving both efficiency and accuracy. By reducing the number of actions required, simplifying complex operations, and enabling the reuse of existing functions in software infrastructures, code agents streamline the process.

![Code vs JSON Actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)

## Why Code Agents?

In a multi-step agent process, the LLM writes and executes actions, often involving external tool calls. Typically, these actions are written in JSON format, specifying tool names and arguments, which the system must parse to determine which tool to execute.

However, research has shown that it's more effective for tool-calling LLMs to work directly with code. This is one of the core ideas behind smolagents, as illustrated in the diagram above from the paper [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030).

Some key advantages of writing actions in code rather than JSON include:

* **Composability**: Easily nest or reuse actions.
* **Object Management**: Store complex structures, such as images, directly in code.
* **Generality**: Code can express any task a computer is capable of performing.
* **Representation in LLM Training Data**: High-quality code is already part of the LLM's training dataset.

## How Does a Code Agent Work?

![From https://huggingface.co/docs/smolagents/conceptual_guides/react](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/codeagent_docs.png)

The diagram above illustrates how `CodeAgent.run()` operates, following the [ReAct framework](https://huggingface.co/papers/2210.03629), which is currently the preferred approach for building multi-step agents. We introduced the `MultiStepAgent` in the previous section, which serves as the core building block for smolagents. Within these agents, we can incorporate a `CodeAgent`, as we will see in this example.

A `CodeAgent` performs actions through a cycle of steps, with existing variables and knowledge being incorporated into the agent’s logs as follows:

1. The system prompt is stored in a `SystemPromptStep`, and the user query is logged in a `TaskStep`.

2. Then, the following while loop is executed:

    2.1 `agent.write_memory_to_messages()` writes the agent's logs into a list of LLM-readable [chat messages](https://huggingface.co/docs/transformers/en/chat_templating).
    
    2.2 These messages are sent to a `Model`, which generates a completion. The completion is parsed to extract the action, which, in our case, could be a code snippet since we’re working with a `CodeAgent`.
    
    2.3 The action is executed.
    
    2.4 The results are logged into memory in an `ActionStep`.

At the end of each step, if the agent includes any function calls (in `agent.step_callback`), they are executed.

## Let's See Some Examples

Now that we understand how a multi-step `CodeAgent` works, let’s look at two examples. In the following scenario, we create a code agent that can search the web using DuckDuckGo.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```

In the following example, we create a code agent that can get the travel time between two locations. Here, we use the `@tool` decorator to define a custom function that can be used as a tool.

```python
from smolagents import CodeAgent, HfApiModel, tool
from typing import Optional

@tool
def get_travel_duration(start_location: str, destination_location: str, departure_time: Optional[int] = None) -> str:
    """Gets the travel time in car between two places.
    
    Args:
        start_location: the place from which you start your ride
        destination_location: the place of arrival
        departure_time: the departure time, provide only a `datetime.datetime` if you want to specify this
    """
    import googlemaps # All imports are placed within the function, to allow for sharing to Hub.
    import os

    gmaps = googlemaps.Client(os.getenv("GMAPS_API_KEY"))

    if departure_time is None:
        from datetime import datetime
        departure_time = datetime(2025, 1, 6, 11, 0)

    directions_result = gmaps.directions(
        start_location,
        destination_location,
        mode="transit",
        departure_time=departure_time
    )
    return directions_result[0]["legs"][0]["duration"]["text"]

agent = CodeAgent(tools=[get_travel_duration], model=HfApiModel(), additional_authorized_imports=["datetime"])

agent.run("Can you give me a nice one-day trip around Paris with a few locations and the times? Could be in the city or outside, but should fit in one day. I'm travelling only via public transportation.")
```

These examples are just the beginning of what you can do with code agents. You can learn more about how to build code agents in the [smolagents documentation](https://huggingface.co/docs/smolagents).

smolagents provides a lightweight framework for building code agents, with a core implementation of approximately 1,000 lines of code. The framework specializes in agents that write and execute Python code snippets, offering sandboxed execution for security. It supports both open-source and proprietary language models, making it adaptable to various development environments.


## Further Reading

- [smolagents Blog](https://huggingface.co/blog/smolagents) - Introduction to smolagents and code interactions
- [smolagents: Building Good Agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Best practices for reliable agents
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) - Agent design principles
