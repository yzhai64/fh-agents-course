![smolagents banner](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png)
# Why Use smolagents

In this module, we will explore the pros and cons of using [smolagents](https://huggingface.co/blog/smolagents), helping you make an informed decision about whether it's the right framework for your needs.

## When to Use an Agentic Framework

An agentic framework is not always needed when building an application around LLMs. They provide flexibility in the workflow to efficiently solve a specific task, but they're not always necessary. Sometimes, predefined workflows are sufficient to fulfill user requests, and there is no real need for an agentic framework. If the approach to build an agent is simple, like a chain of prompts, using plain code may be enough. The advantage is that the developer will have full control and understanding of their system without abstractions.

However, when the workflow becomes more complex, such as letting an LLM call functions or using multiple agents, these abstractions start to become helpful.

Considering these ideas, we can already identify the need for some features:

* An LLM engine powering the system.
* A list of tools the agent can access.
* A parser for extracting tool calls from the LLM output.
* A system prompt synced with the parser.
* A memory system.
* Error logging and retry mechanisms to control LLM mistakes.

## What is `smolagents`?

`smolagents` is a simple yet powerful framework for building AI agents. It provides LLMs with the _agency_ to interact with the real world, such as calling search or image generation tools. AI agents are programs where LLM outputs control the workflow.

### Key Advantages of `smolagents`
- **Simplicity:** Minimal code complexity and abstractions.
- **Flexible LLM Support:** Works with any LLM through integration with Hugging Face tools and external APIs.
- **Code-First Approach:** First-class support for Code Agents that directly write their actions in code.
- **HF Hub Integration:** Seamless integration with Hugging Face Hub, allowing the use of Gradio Spaces as tools.

### Code vs. JSON Actions
Unlike other frameworks where agents write actions in JSON that require parsing, `smolagents` focuses on direct tool calls in code, simplifying the execution process. The following diagram illustrates this difference:

![Code vs. JSON actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)

### Agent Types in `smolagents`
Agents in `smolagents` inherit from [MultiStepAgent](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.MultiStepAgent), enabling them to operate in multiple steps, where each step consists of:
- One thought
- One tool call and execution

There are two types of agents available:
1. **[CodeAgent](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.CodeAgent):** The default agent type that writes tool calls using Python.
2. **[ToolCallingAgent](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.ToolCallingAgent):** Writes tool calls in JSON.

We will explore each agent type in more detail in the following units.

> **_Note_**: In smolagents, tools are defined using `@tool` or the `Tool` class. They are distinct from `ToolCallingAgent`. Both `CodeAgents` and `ToolCallingAgent` utilize tools. Keep this distinction in mind throughout the rest of the unit to avoid confusion!

### Model Integration in `smolagents`
`smolagents` supports flexible LLM integration, allowing you to use any callable model as long as it meets [certain criteria](https://huggingface.co/docs/smolagents/main/en/reference/models). To simplify connections with various model types, the framework provides predefined classes:

- **[TransformersModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.TransformersModel):** Implements a local `transformers` pipeline for seamless integration.
- **[HfApiModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.HfApiModel):** Wraps Hugging Face's [InferenceClient](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference) to support the [Inference API](https://huggingface.co/docs/api-inference/index) and [Inference Providers](https://huggingface.co/blog/inference-providers).
- **[LiteLLMModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.LiteLLMModel):** Leverages [LitLLM](https://www.litellm.ai/) for lightweight model interactions.
- **[OpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.OpenAIServerModel):** Connects to models compatible with the OpenAI API server.
- **[AzureOpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.AzureOpenAIServerModel):** Supports integration with any Azure OpenAI deployment.

This flexibility ensures that developers can choose the most suitable model integration for their specific use cases.

## When to use smolagents

All the previously mentioned functionality is available in `smolagents`. So the question becomes: when should we use smolagents instead of other possible frameworks? 

You may choose smolagents if:
- You prioritize a lightweight and minimal solution.
- You would like to experiment without complex configurations.
- The logic for your application is straightforward.

Join us as we dive deep into smolagents in the upcoming sections!

## Further Reading

- [smolagents Blog](https://huggingface.co/blog/smolagents) - Introduction to smolagents and code interactions