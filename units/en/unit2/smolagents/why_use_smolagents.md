# Why Use smolagents

In this module, we will explore the pros and cons of using [smolagents](https://huggingface.co/blog/smolagents), helping you make an informed decision about whether it's the right framework for your needs.

![Agentic pipeline](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif)

## When to Use an Agentic Framework

An agentic framework is not always needed when building an application around LLMs. They provide flexibility in the workflow to efficiently solve a specific task, but they're not always necessary. Sometimes, predefined workflows are sufficient to fulfill user requests, and there is no real need for an agentic framework. Even if the agentic approach is low-level, like chains or routers, using plain code may be enough since the developer will have full control and understanding of the system.

However, when the workflow becomes more complex, such as letting an LLM call functions (tool calling) or execute multiple steps (multi-step agent), these abstractions start to become necessary.

Considering these ideas, we can already identify the need for some features:

* An LLM engine powering the system.
* A list of tools the agent can access.
* A parser for extracting tool calls from the LLM output.
* A system prompt synced with the parser.
* A memory system.
* Error logging and retry mechanisms to control LLM mistakes.

## What is `smolagents`?

`smolagents` is the simplest framework to build powerful agents. It provides the LLM the _agency_ to access the real world, for example, by calling a search or image generation tool. AI agents are programs where LLM outputs control the workflow.

smolagents offers several advantages over other frameworks:

* Simple in terms of code complexity with minimal abstractions.
* Support for any LLM through integration with Hugging Face's tools and external tools.
* First-class support for Code Agents, which write their actions in code.
* Integration with HF Hub, which, for example, allows using Gradio Spaces as tools.

It also offers the features introduced in the previous section. Code agents are a key component of this framework. For some LLMs agentds, they use JSON for writing the actions that the agent needs to execute, which must be parsed for execution. In comparison, smolagents directly aim for tool calling in code. You can refer to the diagram below for a better understanding of this concept:

![Code vs. JSON actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)

## When to use smolagents

All the previously mentioned functionality is available in `smolagents`. So the question becomes: when should we use smolagents instead of other possible frameworks? 

You may choose smolagents if:
- You prioritize a lightweight and minimal solution.
- You would like to experiment without complex configurations.
- The logic for your application is straightforward.
