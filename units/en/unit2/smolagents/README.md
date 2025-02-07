# smolagents

![smolagents license to call](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png)

This module covers how to build effective agents using the [`smolagents`](https://github.com/huggingface/smolagents) library, which provides a lightweight framework for creating capable AI agents.

## Module Overview

This module provides a comprehensive overview of key concepts and practical strategies for building intelligent agents using smolagents. With so many open-source frameworks available, it's essential to understand the components and capabilities that make smolagents a valuable option—or to determine when another solution might be a better fit. We'll explore critical agent types, including retrieval agents that access and synthesize information, code agents designed for software development tasks, and custom function agents for creating modular, function-driven workflows. Additionally, we'll cover the integration of vision capabilities and web browsing to unlock new possibilities for dynamic and context-aware applications.

## Contents

### 1️⃣ [Why Use smolagents](./why_use_smolagents.md)

`smolagents` is one of the many open-source agent frameworks available for application development. Alternative options include `LangChain`, `LlamaIndex` and `LangGraph`, which are also covered in other modules in this course. This library offers several key features that might make it a great fit for specific use cases, but there are also some limitations that may lead you to consider other solutions. This module explores the advantages and drawbacks of using `smolagents`, helping you make an informed decision based on your project's requirements.

### 2️⃣ [Retrieval Agents](./retrieval_agents.md)

Retrieval agents combine models with knowledge bases. These agents can search and synthesize information from multiple sources, leveraging vector stores for efficient retrieval and implementing Retrieval Augmented Generation (RAG) patterns. They are great at combining web search with custom knowledge bases while maintaining conversation context through memory systems. The module covers implementation strategies including fallback mechanisms for robust information retrieval.

### 3️⃣ [Code Agents](./code_agents.md)

Code agents are specialized autonomous systems designed for software development tasks. These agents excel at analyzing and generating code, performing automated refactoring, and integrating with development tools. The module covers best practices for building code-focused agents that can understand programming languages, work with build systems, and interact with version control while maintaining high code quality standards.

### 4️⃣ [Custom Functions](./custom_functions.md)

Custom function agents extend basic AI capabilities through specialized function calls. This module explores how to design modular and extensible function interfaces that integrate directly with your application's logic. You'll learn to implement proper validation and error handling while creating reliable function-driven workflows. The focus is on building simple systems where agents can predictably interact with external tools and services.

### 5️⃣ [Vision Agents](./vision_agents.md)

Vision agents enhance traditional agent capabilities by integrating vision as a new modality. This opens up opportunities to combine Vision-Language Models (VLMs) within agent pipelines, enabling applications that can process and interpret visual information. In this module, you'll learn how to design and implement agentic pipelines that incorporate VLMs, unlocking advanced functionalities such as image-based reasoning, visual data analysis, and multimodal interactions.

### 6️⃣ [Browser Agents](./browser_agents.md)

Browser agents are a powerful application of vision agents, designed to enable web browsing with visual capabilities. By integrating vision models, these agents can interact with web content in new ways, such as interpreting visual elements, extracting relevant information from images or videos, and navigating web pages more effectively. This module explores how browser agents open up a wide range of innovative use cases, from dynamic content extraction to visual web automation. 


### Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Building a Research Agent | Create an agent that can perform research tasks using retrieval and custom functions | 🐢 Build a simple RAG agent <br> 🐕 Add custom search functions <br> 🦁 Create a full research assistant | [Notebook](./notebooks/agents.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/8_agents/notebooks/agents.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Resources

- [smolagents Documentation](https://huggingface.co/docs/smolagents) - Official docs for the smolagents library
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Research paper on agent architectures
- [Agent Guidelines](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Best practices for building reliable agents
- [LangChain Agents](https://python.langchain.com/docs/how_to/#agents) - Additional examples of agent implementations
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) - Understanding function calling in LLMs
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Guide to implementing effective RAG
