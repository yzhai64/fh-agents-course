![smolagents banner](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png)
# Why use smolagents

In this module, we will explore the pros and cons of using [smolagents](https://huggingface.co/blog/smolagents), helping you make an informed decision about whether it's the right framework for your needs.

## What is `smolagents`?

`smolagents` is a simple yet powerful framework for building AI agents. It provides LLMs with the _agency_ to interact with the real world, such as searching or generating images.

As we learned in unit 1, AI agents are programs that use LLMs to generate **'thoughts'** based on **'observations'** to perform **'actions'**. Let's explore how this is implemented in smolagents.

### Key Advantages of `smolagents`
- **Simplicity:** Minimal code complexity and abstractions
- **Flexible LLM Support:** Works with any LLM through integration with Hugging Face tools and external APIs
- **Code-First Approach:** First-class support for Code Agents that directly write their actions in code
- **HF Hub Integration:** Seamless integration with Hugging Face Hub, allowing the use of Gradio Spaces as tools

### When to use smolagents

With these advantages in mind, when should we use smolagents over other frameworks? 

Smolagents is ideal when:
- You need a lightweight and minimal solution
- You want to experiment quickly without complex configurations
- Your application logic is straightforward

### Code vs. JSON Actions
Unlike other frameworks where agents write actions in JSON that require parsing, `smolagents` focuses on direct tool calls in code, simplifying the execution process. The following diagram illustrates this difference:

![Code vs. JSON actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)

### Agent Types in `smolagents`
Agents in `smolagents` operate as multi-step agents. Each [`MultiStepAgent`](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.MultiStepAgent) performs:
- One thought
- One tool call and execution

The framework offers two types of agents:
1. **[CodeAgent](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.CodeAgent):** The default agent type that writes tool calls using Python
2. **[ToolCallingAgent](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.ToolCallingAgent):** Writes tool calls in JSON

We will explore each agent type in more detail in the following units.

<Tip> 
In smolagents, tools are defined using <code>@tool</code> or the <code>Tool</code> class. They are distinct from <code>ToolCallingAgent</code>. Both <code>CodeAgents</code> and <code>ToolCallingAgent</code> utilize tools. Keep this distinction in mind throughout the rest of the unit to avoid confusion! 
</Tip>

### Model Integration in `smolagents`
`smolagents` supports flexible LLM integration, allowing you to use any callable model that meets [certain criteria](https://huggingface.co/docs/smolagents/main/en/reference/models). The framework provides several predefined classes to simplify model connections:

- **[TransformersModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.TransformersModel):** Implements a local `transformers` pipeline for seamless integration.
- **[HfApiModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.HfApiModel):** Wraps Hugging Face's [InferenceClient](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference) to support the [Inference API](https://huggingface.co/docs/api-inference/index) and [Inference Providers](https://huggingface.co/blog/inference-providers).
- **[LiteLLMModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.LiteLLMModel):** Leverages [LitLLM](https://www.litellm.ai/) for lightweight model interactions.
- **[OpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.OpenAIServerModel):** Connects to models compatible with the OpenAI API server.
- **[AzureOpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.AzureOpenAIServerModel):** Supports integration with any Azure OpenAI deployment.

This flexibility ensures that developers can choose the most suitable model integration for their specific use cases.

Join us as we dive deep into smolagents in the upcoming sections!

## Resources

- [smolagents Blog](https://huggingface.co/blog/smolagents) - Introduction to smolagents and code interactions