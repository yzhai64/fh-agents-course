# Building Agents That Use Code

Code agents are the default agent type in smolagents. They generate Python tool calls directly to perform actions, making them both efficient and accurate. Their streamlined approach reduces the number of required actions, simplifies complex operations, and enables reuse of existing code functions. smolagents provides a lightweight framework for building code agents, implemented in approximately 1,000 lines of code.

![Code vs JSON Actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)
Graphic from the paper [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030)

<Tip>
If you want to learn more about why code agents are effective, check out <a href="https://huggingface.co/docs/smolagents/en/conceptual_guides/intro_agents#code-agents" target="_blank">this guide</a> from the smolagents documentation. 
</Tip>

## Why Code Agents?

In a multi-step agent process, the LLM writes and executes actions, typically involving external tool calls. Traditional approaches use JSON format to specify tool names and arguments as strings, which the system must parse to determine which tool to execute.

However, research shows that tool-calling LLMs work more effectively with code directly. This is a core principle of smolagents, as shown in the diagram above from [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030).

Writing actions in code rather than JSON offers several key advantages:

* **Composability**: Easily combine and reuse actions
* **Object Management**: Work directly with complex structures like images
* **Generality**: Express any computationally possible task
* **Natural for LLMs**: High-quality code is already present in LLM training data

## How Does a Code Agent Work?

![From https://huggingface.co/docs/smolagents/conceptual_guides/react](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/codeagent_docs.png)

The diagram above illustrates how `CodeAgent.run()` operates, following the [ReAct framework](https://huggingface.co/papers/2210.03629), which is currently the preferred approach for building multi-step agents. We introduced the `MultiStepAgent` in the previous section, which serves as the core building block for smolagents. Within these agents, we can incorporate a `CodeAgent`, as we will see in this example.

A `CodeAgent` performs actions through a cycle of steps, with existing variables and knowledge being incorporated into the agent's logs as follows:

1. The system prompt is stored in a `SystemPromptStep`, and the user query is logged in a `TaskStep`.

2. Then, the following while loop is executed:

    2.1 `agent.write_memory_to_messages()` writes the agent's logs into a list of LLM-readable [chat messages](https://huggingface.co/docs/transformers/en/chat_templating).
    
    2.2 These messages are sent to a `Model`, which generates a completion. The completion is parsed to extract the action, which, in our case, could be a code snippet since we're working with a `CodeAgent`.
    
    2.3 The action is executed.
    
    2.4 The results are logged into memory in an `ActionStep`.

At the end of each step, if the agent includes any function calls (in `agent.step_callback`), they are executed. Below, you can see an comparison diagram between a multi-step agent using the ReAct framework and a one step agent.

![Comparison diagram between a multi-step agent using the ReAct framework and a one step agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)

## Let's See Some Examples

Alfred is planning a party at the Wayne family mansion and needs your help to ensure everything goes smoothly. To assist him, we'll apply what we've learned about how a multi-step `CodeAgent` operates.

If you haven't installed `smolagents` yet, you can do so by running the following command:

```bash
pip install smolagents
```

Let's also login to the HF Hub to have access to the Inference API.

```python
from huggingface_hub import notebook_login

notebook_login()
```

### Selecting a Playlist for the Party Using `smolagents`

An important part of a successful party is the music. Alfred needs some help selecting the playlist, and we're covered using `smolagents`. We can build an agent capable of searching the web using DuckDuckGo. To give the agent access to this tool, we include it in the tool list when creating the agent.

For the model, we'll rely on `HfApiModel`, which provides access to Hugging Face's [Inference API](https://huggingface.co/docs/api-inference/index). The default model is `"Qwen/Qwen2.5-Coder-32B-Instruct"`, which is performant and available for fast inference, but you can select any compatible model from the hub.

Running an agent is quite straightforward:

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
```

When you run this example, the output will display a trace of the workflow steps being executed. It will also print the corresponding Python code with the message: 

```python
 ─ Executing parsed code: ──────────────────────────────────────────────────────────────────────────────────────── 
  results = web_search(query="best music for a Batman party")                                                      
  print(results)                                                                                                   
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
```

After a few steps, you'll see the generated playlist that Alfred can use for the party! 🎵

### Using a Custom Tool to Prepare the Menu

Now that we have selected a playlist, we need to organize the menu for the guests. Again, Alfred can take advantage of smolagents to do so. Here, we use the `@tool` decorator to define a custom function that acts as a tool. We'll cover tool creation in more detail later, so for now, we can simply run the code.

As you can see, the generated tool is included in the `tools` list.

```python
from smolagents import CodeAgent, tool

# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

# Alfred, the butler, preparing the menu for the party
agent = CodeAgent(tools=[suggest_menu], model=HfApiModel())

# Preparing the menu for the party
agent.run("Prepare a formal menu for the party.")
```

The agent will run for a few steps until finding the answer.

The menu is ready! 🥗

### Using Python Imports Inside the Agent

We have the playlist and menu ready, but we need to check one more crucial detail: preparation time!

Alfred needs to calculate when everything would be ready if he started preparing now, in case they need assistance from other superheroes.

When creating the agent, we use `additional_authorized_imports` to allow the datetime module. Code execution has strict security measures - imports outside a predefined safe list are blocked by default. However, you can authorize additional imports by passing them as strings in `additional_authorized_imports`.

For more details on secure code execution, see the official [guide](https://huggingface.co/docs/smolagents/tutorials/secure_code_execution).

```python
from smolagents import CodeAgent, HfApiModel
import numpy as np
import time
import datetime

agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['datetime'])

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    3. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)
```

These examples are just the beginning of what you can do with code agents, and we're already starting to see their utility for preparing the party. You can learn more about how to build code agents in the [smolagents documentation](https://huggingface.co/docs/smolagents).

smolagents specializes in agents that write and execute Python code snippets, offering sandboxed execution for security. It supports both open-source and proprietary language models, making it adaptable to various development environments.

### Sharing Our Custom Party Preparator Agent to the Hub

Wouldn't it be amazing to share the Alfred agent with the community? By doing so, anyone can easily download and use the agent directly from the Hub, bringing the ultimate party preparator of Gotham to their fingertips! Let's make it happen! 🎉

The `smolagents` library makes this possible by allowing you to share a complete agent with the community and download others for immediate use. It's as simple as the following:


```python
# Change to your username and repo name
agent.push_to_hub('sergiopaniego/AlfredAgent')
```

To download the agent, use this:

```python
# Change to your username and repo name
alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent')

alfred_agent.run("Give me best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")
```

What's also exciting is that shared agents are directly available as HF Spaces, allowing you to interact with them in real-time. You can explore other agents [here](https://huggingface.co/spaces/davidberenstein1957/smolagents-and-tools).

For example, the _AlfredAgent_ is available [here](https://huggingface.co/spaces/sergiopaniego/AlfredAgent). You can try it out directly below:

<iframe
	src="https://sergiopaniego-alfredagent.hf.space/"
	frameborder="0"
	width="850"
	height="450"
></iframe>

Now, you may be wondering—how did Alfred build such an agent using `smolagents`? By integrating several tools, he can generate an agent as follows. Don't worry about the tools for now, as we'll have a dedicated section later in this unit to explore that in detail:

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, VisitWebpageTool, FinalAnswerTool, Tool, tool

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.
    
    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    
    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)
    
    return best_service

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""
    
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }
    
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        
        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")


# Alfred, the butler, preparing the menu for the party
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(), 
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool()
        ], 
    model=HfApiModel(),
    max_steps=10,
    verbosity_level=2
)

agent.run("Give me best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")
```

As you can see, we've created a `CodeAgent` with several tools that enhance the agent's functionality, turning it into the ultimate party preparator ready to share with the community! 🎉

As a challenge, it's your turn: build your own agent and share it with the community using the knowledge we've just learned! 🕵️‍♂️💡

## Resources

- [smolagents Blog](https://huggingface.co/blog/smolagents) - Introduction to smolagents and code interactions
- [smolagents: Building Good Agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Best practices for reliable agents
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) - Agent design principles
