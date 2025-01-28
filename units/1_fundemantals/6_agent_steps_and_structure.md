# Understanding AI Agents through the Thought-Action-Observation Cycle

AI agents are systems that can reason, plan, and interact with their environment. To grasp how they function, imagine a continuous cycle of thinking, acting, and observing. This dynamic process allows agents to retrieve information, make decisions, and take actions, to navigate their environment and achieve their goals.

## The Core Components

This course will focus on three core components of AI agents. We will explore how they work together to form a cohesive system and refer back to these components throughout the course. In this section, we will explore components conceptually and in the following sections we focus on implementing them.

## The Thought-Action-Observation Cycle

The three components work together in a continuous loop:

1. **Observation:** The agent observes its environment
2. **Thought:** The agent processes the observation and decides what to do
3. **Action:** The agent takes action based on its thoughts
4. **Observation:** The agent observes the results, and the cycle continues

### Thoughts

Thoughts represent the agent's internal reasoning and planning processes. This utilises the agent's Large Language Model (LLM) to analyze information. Think of it as the agent's internal dialogue, where it considers the task at hand and strategizes its approach.

The agent's thoughts are responsible for analyzing observations, deciding on actions, formulating plans, and engaging in multi-step reasoning. Through this process, the agent can break down complex problems into smaller, more manageable steps, reflect on past experiences, and continuously adjust its plans based on new information.

Below are some examples of the types of thoughts an agent might have:

| Type of Thought | Example |
|----------------|---------|
| Planning | "I need to break this task into three steps: 1) gather data, 2) analyze trends, 3) generate report" |
| Analysis | "Based on the error message, the issue appears to be with the database connection parameters" |
| Decision Making | "Given the user's budget constraints, I should recommend the mid-tier option" |
| Problem Solving | "To optimize this code, I should first profile it to identify bottlenecks" |
| Memory Integration | "The user mentioned their preference for Python earlier, so I'll provide examples in Python" |
| Self-Reflection | "My last approach didn't work well, I should try a different strategy" |
| Goal Setting | "To complete this task, I need to first establish the acceptance criteria" |
| Prioritization | "The security vulnerability should be addressed before adding new features" |


### Actions

Actions are the concrete steps an AI agent takes to interact with its environment. These can range from browsing the web for information to controlling a robot in a physical space. Consider an agent assisting with customer service - it might retrieve customer data, offer support articles, or escalate issues to human representatives.

| Type of Action | Description |
|----------------|-------------|
| Information Gathering | Performing web searches, querying databases, retrieving documents |
| Tool Usage | Making API calls, running calculations, writing and executing code |
| Environment Interaction | Manipulating digital interfaces, controlling physical robots or devices |
| Communication | Engaging with users through chat, collaborating with other AI agents |

### Observations

Observations are how an AI agent perceives the world and the consequences of its actions. They provide crucial information that fuels the agent's thought process and guides future actions. These observations can take many forms, from reading webpage text to monitoring a robot arm's position.

| Type of Observation | Example |
|-------------------|----------|
| System Feedback | Error messages, success notifications, status codes |
| Data Changes | Database updates, file modifications, state changes |
| Environmental Data | Sensor readings, system metrics, resource usage |
| Response Analysis | API responses, query results, computation outputs |
| Time-based Events | Deadlines reached, scheduled tasks completed |

## Specialized Agents: Code vs. JSON

<!-- TODO: @Jofthomas -->