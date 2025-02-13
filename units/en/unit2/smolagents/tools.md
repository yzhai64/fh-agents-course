# Tools  

As we learnt in unit 1, agents use tools to perform actions. In `smolagents` tools are treated as functions that an LLM can use within an agentic system. To interact with a tool, the LLM requires an **interface description** with the following components:  

- **Name**  
- **Tool description**  
- **Input types and their descriptions**  
- **Output type**  

For complex tools, we can implement a class instead of a Python function. The class wraps the function with metadata that helps the LLM understand how to use it effectively.  

Below, you can see an animation illustrating how a tool call is managed:  

![Agentic pipeline from https://huggingface.co/docs/smolagents/conceptual_guides/react](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif)  

## General Structure  

In `smolagents`, tools can be defined in two ways:  
1. **Creating a subclass of `Tool`**, which provides useful methods.  
2. **Using the `@tool` decorator** to define a function-based tool.  

### Defining a Tool as a Python Class  

The first approach involves creating a subclass of [`Tool`](https://huggingface.co/docs/smolagents/v1.8.1/en/reference/tools#smolagents.Tool). In this class, we define:  

- `name`: The tool’s name.  
- `description`: A description used to populate the agent's system prompt.  
- `inputs`: A dictionary with keys `type` and `description`, providing information to help the Python interpreter process inputs.  
- `output_type`: Specifies the expected output type.  
- `forward`: The method containing the inference logic to execute.

Below, we can see an example of a tool built using `Tool` and to integrate it within a `CodeAgent`.

```python
from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        from huggingface_hub import list_models

        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

model_downloads_tool = HFModelDownloadsTool()

agent = CodeAgent(tools=[model_downloads_tool], model=HfApiModel())

agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```

### `@tool` Decorator  

The `@tool` decorator is the recommended way to define simple tools. Under the hood, smolagents will parse basic information about the function from Python. So if you name you function clearly and a good docstring, it will be easier for the LLM to use. Using this approach, we define a function with:  

- **A clear and descriptive function name** that helps the LLM understand its purpose.  
- **Type hints for both inputs and outputs** to ensure proper usage.  
- **A detailed description**, including an `Args:` section where each argument is explicitly described. These descriptions provide valuable context for the LLM, so it's important to write them carefully.  

Below is an example of a function using the `@tool` decorator, replicating the same functionality as the previous example:

```python
from smolagents import tool

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

## Default Toolbox  

`smolagents` comes with a set of pre-built tools that can be directly injected into your agent. The [default toolbox](https://huggingface.co/docs/smolagents/guided_tour?build-a-tool=Decorate+a+function+with+%40tool#default-toolbox) includes:  

- **PythonInterpreterTool**  
- **FinalAnswerTool**  
- **UserInputTool**  
- **DuckDuckGoSearchTool**  
- **GoogleSearchTool**  
- **VisitWebpageTool**  


## Sharing and Importing Tools

One of the most powerful features of **smolagents** is the ability to share your custom tools to the Hub, as well as load tools shared by the community. This includes integrating **HF Spaces** or **LangChain tools**. Below are examples showcasing each of these functionalities:

### Sharing a Tool to the Hub

To share your custom tool, you can upload it to your Hugging Face account using the `push_to_hub()` method:

```python
model_downloads_tool.push_to_hub("{your_username}/hf-model-downloads", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```

### Importing a Tool from the Hub

You can import tools developed by other users by utilizing the load_tool() function:

```python
from smolagents import load_tool, CodeAgent

model_download_tool = load_tool(
    "{your_username}/hf-model-downloads", # m-ric/text-to-image
    trust_remote_code=True
)
```

### Importing a Hugging Face Space as a Tool

You can also import a HF Space as a tool using `Tool.from_space()`. This opens up possibilities for integrating with thousands of spaces from the community for tasks from image generation to data analysis. The tool will connect with the spaces Gradio backend using the `gradio_client`, so make sure to install it via `pip` if you don't have it already:

```python
from smolagents import CodeAgent, HfApiModel, Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "Improve this prompt, then generate an image of it.", additional_args={'user_prompt': 'A rabbit wearing a space suit'}
)
```

### Importing a LangChain Tool

You can also load tools from LangChain using the `Tool.from_langchain()` method. Here's how to import and use a LangChain tool:

```python
from langchain.agents import load_tools, Tool

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("How many more blocks (also denoted as layers) are in BERT base encoder compared to the encoder from the architecture proposed in Attention is All You Need?")
```

## Further Reading

- [Tools Tutorial](https://huggingface.co/docs/smolagents/tutorials/tools) - Explore this tutorial to learn how to work with tools effectively.
- [Tools Documentation](https://huggingface.co/docs/smolagents/v1.8.1/en/reference/tools) - Comprehensive reference documentation on tools.
- [Tools Guided Tour](https://huggingface.co/docs/smolagents/v1.8.1/en/guided_tour#tools) - A step-by-step guided tour to help you build and utilize tools efficiently.
- [Building Effective Agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - A detailed guide on best practices for developing reliable and high-performance custom function agents.