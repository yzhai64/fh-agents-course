# Building Agents That Use Code

Code agents are the default agent type in smolagents. They generate their own Python tool calls to perform actions, improving both efficiency and accuracy. Code agents are really streamlined because they reduce the number of actions required, simplify complex operations, and enable the reuse of existing functions from code. We'll build on these advantages throughout this page. smolagents provides a lightweight framework for building code agents, with a core implementation of approximately 1,000 lines of code.

![Code vs JSON Actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)
Graphic from the paper [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030)

<Tip>If you want to learn more about why code agents are effective, check out <a href="https://huggingface.co/docs/smolagents/en/conceptual_guides/intro_agents#code-agents" target="_blank">this guide</a> from the smolagents documentation. </Tip>

## Why Code Agents?

In a multi-step agent process, the LLM writes and executes actions, often involving external tool calls. Without code agents, these actions are written in JSON format, specifying tool names and arguments as strings, which the system must parse to determine which tool to execute.

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

At the end of each step, if the agent includes any function calls (in `agent.step_callback`), they are executed. Below, you can see an comparison diagram between a multi-step agent using the ReAct framework and a one step agent.

![Comparison diagram between a multi-step agent using the ReAct framework and a one step agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)

## Let's See Some Examples

Now that we understand how a multi-step `CodeAgent` works, let's explore three examples. If you haven't install smolagents yet, you can run the following command:

```bash
pip install smolagents
```

In the first scenario, we create a code agent capable of searching the web using DuckDuckGo. To grant the agent access to this tool, we include it in the tool list when creating the agent.  

For the model, we'll rely on `HfApiModel`, which provides access to Hugging Face's [Inference API](https://huggingface.co/docs/api-inference/index).  The default model is `"Qwen/Qwen2.5-Coder-32B-Instruct"` which is performant and available for fast inference, but you can select any compatible model from the hub.

Running an agent is quite straightforward:

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```

When you run this example, the output will display a trace of the workflow steps being executed. It will also print the corresponding Python code with the message:  

```python
Executing parsed code:...
```

After a few steps, you'll likely see the final answer!

In the following example, we create a code agent that check which is the most downloaded 'text-to-video' model on the Hugging Face Hug. Here, we use the `@tool` decorator to define a custom function that acts as a tool. We'll cover tool creation in more detail later, so for now, we can simply run the code.  

As you can see, the generated tool is included in the `tools` list. 

```python
from smolagents import CodeAgent, HfApiModel, tool
from huggingface_hub import list_models

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id

agent = CodeAgent(tools=[model_download_tool], model=HfApiModel())

agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```

The agent will run for a few steps until finding the answer.

For the final example, we would like to check the title of a webpage. As you can see, when creating the agent, we use `additional_authorized_imports`. Code execution enforces strict security measures, meaning that imports outside a predefined safe list are not allowed by default. However, developers can authorize additional imports by passing them as a list of strings in `additional_authorized_imports`. In this case, we need to import `requests` for requesting the url and `bs4` for scraping information from web pages.  

For more details on secure code execution, check out the official [guide](https://huggingface.co/docs/smolagents/tutorials/secure_code_execution).

```python
from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['requests', 'bs4'])

agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

These examples are just the beginning of what you can do with code agents. You can learn more about how to build code agents in the [smolagents documentation](https://huggingface.co/docs/smolagents).

smolagents specializes in agents that write and execute Python code snippets, offering sandboxed execution for security. It supports both open-source and proprietary language models, making it adaptable to various development environments.

## Further Reading

- [smolagents Blog](https://huggingface.co/blog/smolagents) - Introduction to smolagents and code interactions
- [smolagents: Building Good Agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Best practices for reliable agents
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) - Agent design principles
