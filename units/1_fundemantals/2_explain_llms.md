# Explain Large Language Models

<!-- Explanation of LLMs, including the family tree of models: encoders, seq2seq, decoders. Decoders are autoregressive and continue until EOS.  -->

<!-- TODO: @burtenshaw -->

In this section we will explore the basics of Large Language Models (LLMs) and how they are used within AI Agents.

## What is a Large Language Model?

A Large Language Model (LLM) is a type of artificial intelligence model that excels at understanding and generating human language. They are trained on vast amounts of text data, allowing them to learn patterns, nuances, and structure in language. These models typically consist of billions or even trillions of parameters, enabling them to capture intricate linguistic patterns and perform complex tasks.

They are primarily text-based models, but there are variants for other modalities like images, audio, and video. Through language, LLMs can also use tools to perform tasks, such as searching the web, performing calculations, or even controlling physical robots.   

## Key Characteristics

- **Scale**: Because of their massive size, LLMs can store rich linguistic knowledge and perform complex, multi-step tasks.  
- **Generative**: They can write creative content, translate languages, or generate code.  
- **Contextual Understanding**: They can maintain context over long conversations, allowing for more natural and engaging interactions.  

## How do LLMs learn?

LLMs are trained on large datasets of text, where they learn to predict the next word in a sequence through a self-supervised or masked language modeling objective. From this unsupervised learning, the model learns the structure of the language and underlying patterns in text.

Following this, LLMs can be fine-tuned on a supervised learning objective to perform specific tasks. For example, some are trained for conversational structures or tool usage, while others focus on classification or code generation.

Most LLMs use the Transformer architecture, which is a type of neural network that can store and handle complex patterns in language data. The most common type of Transformer for LLMs is the decoder, an autoregressive algorithm that continues generating text until a special stop token is reached. We will explore special tokens in more detail in the [next section](./3_messages_and_special_tokens.md).

## How can I use LLMs?

You can run LLMs locally on your own laptop (if you have sufficient hardware) or call them through an API. Some models require a lot of memory to run efficiently, so you'll need to factor in these hardware requirements when choosing a model.

During this course, we will initially use models through APIs on the Hugging Face hub. Later, we will explore how to run models locally on your own hardware.

## How are LLMs used in AI Agents?

LLMs are a key component of AI Agents, providing the foundation for understanding and generating human language. They can interpret user instructions, maintain context in conversations, and decide which tools to use. We will explore these steps in more detail in [dedicated sections](./6_agent_steps_and_structure.md).

Some common use cases include:
- **Conversational AI**: Chatbots and virtual assistants responding to user queries.  
- **Content Generation**: Articles, stories, or creative writing.  
- **Code Generation**: Helping developers with code completion and snippet suggestions.  
- **Question Answering**: Delivering detailed and coherent answers on a wide range of topics.  
- **Translation**: Converting text between different languages with high accuracy.  

## Challenges and Limitations

While LLMs are powerful, they can sometimes generate biased or factually incorrect content. They may also require substantial computational resources to train and deploy. Researchers are actively exploring methods to reduce these issues and make LLMs more reliable.