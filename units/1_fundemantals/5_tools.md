# Tools
<!-- Discussion on Pydantic tools, conversion to string in the system prompt, and other common tool formats. -->
<!-- TODO: @jofthomas -->

One crucial aspect of AI agents is their ability to utilize tools to enhance their capabilities and accomplish a wider range of tasks. AI agents can be even more effective when combined with other tools. This page provides an overview of AI tools and how they can be integrated into AI agents.

AI agents often rely on tools to extend their capabilities. These tools allow agents to interact with external environments to perform certain tasks. These tasks might include extracting information from databases, querying, coding, and anything else the agent needs to function. When an AI agent uses these tools, it follows specific workflows to carry out tasks, gather observations, or collect the information needed to complete subtasks and fulfill user requests.

## What are AI tools?

Tools are executable processes or external APIs that the agent can use to perform specific tasks. Tools allow agents to interact with external environments to perform certain tasks. These tasks might include extracting information from databases, querying, coding, and anything else the agent needs to do to complete a task.

| Tool Category | Description | Examples |
|---|---|---|
| **Data Retrieval** | Accessing and retrieving information from sources like databases or conversational memory. | Retrieving data from enterprise systems, retrieving information from internal knowledge bases, extracting text from images (OCR) |
| **Code Generation and Execution** | Generating and executing code in different programming languages. | Executing code to perform a calculation. |
| **API Interaction** | Connecting to external APIs to access various services. | Accessing fresh data via APIs (e.g., financial APIs, weather APIs) |
| **General Purpose Tools** | Tools for common tasks and calculations. | Calculators, search engines, or file operations |
| **Specialized Tools** | Tools designed for specific domains or tasks. | Custom scripts |

## Interface Design for Tools

The design of the interface through which an agent uses tools can affect the agent's performance. For example, a search tool that returns results **ordered by relevance** may be more helpful to an AI agent than one that returns results **ordered by frequency**. The interface should be designed to be clear and concise, so that the agent can easily understand how to use the tools. It should also be designed to be flexible, so that the agent can use the tools in different ways depending on the task at hand.

Here are some examples of how interface design can affect an AI agent's performance:

*   **Structured Output:** Providing a specific format or schema for the AI to follow in its response can help the agent to use tools more effectively. For example, if the agent is using a tool to retrieve data from a database, the interface should specify the format in which the data should be returned.
*   **Tool Descriptions:** Providing clear and concise descriptions of the tools can help the agent to understand how to use them. The descriptions should include information about the tool's purpose, its inputs and outputs, and any limitations.
*   **Tool Selection:** The interface should make it easy for the agent to select the appropriate tool for the task at hand. This could involve providing a list of tools or a search function.
*   **Feedback:** The interface should provide feedback to the agent on the results of its tool use. This could involve displaying the output of the tool or providing an error message if the tool fails.

## Tool Use as Action

The use of tools is considered a form of "acting" by an AI agent in an environment. Agents can generate special tokens to invoke tool calls. This "acting" can be guided by "reasoning," which allows the agent to plan and re-plan based on the information gained from the tool. For example, an agent might use a search engine to find information and then use a calculator to perform calculations based on that information. The agent may revise its plan based on the result of the calculation and return the retrieved information. This kind of reasoning-based tool use is the cornerstone of agents, and we will explore it in more detail in the coming sections.

## Conclusion

AI tools play a crucial role in enhancing the capabilities of AI agents. By effectively utilizing these tools, AI agents can perform complex tasks, reason through problems, and interact with users in a more sophisticated and dynamic manner. As the field of AI agents continues to evolve, we can expect to see even more innovative and powerful tools being developed and integrated into these systems.

## Pop Quiz 🍾

1. Which of the following best describes an AI tool?

   A. A process that only generates text responses  
   B. An executable process or external API that allows agents to perform specific tasks and interact with external environments  
   C. A feature that stores agent conversations  

2. Which interface design principle enhances an AI agent's ability to use tools effectively?

   A. Displaying all tool outputs in random formats  
   B. Providing clear descriptions, structured output formats, and feedback mechanisms  
   C. Limiting tool descriptions to technical specifications only  

3. How do AI agents use tools as a form of "acting" in an environment?

   A. By passively waiting for user instructions  
   B. By only using pre-programmed responses  
   C. By generating tokens to invoke tools and revising plans based on the information gained  

---
<details>
<summary>Answer Key (click to reveal)</summary>

1. B. An executable process or external API that allows agents to perform specific tasks and interact with external environments

   • Explanation: The text defines tools as executable processes or external APIs that agents use to perform specific tasks and interact with external environments.

2. B. Providing clear descriptions, structured output formats, and feedback mechanisms

   • Explanation: The interface design section emphasizes the importance of structured output, clear tool descriptions, tool selection, and feedback mechanisms.

3. C. By generating tokens to invoke tools and revising plans based on the information gained

   • Explanation: The text explains that agents can generate special tokens to invoke tools and use reasoning to plan and re-plan based on the information gained.

</details>

