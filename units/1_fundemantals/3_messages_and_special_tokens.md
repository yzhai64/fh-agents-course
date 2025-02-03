# Messages and Special Tokens

<!-- Explanation of messages, special tokens, and chat-template usage. Special tokens are different for every model and allow segmentation of generation in messages. Can go from messages to prompt with the chat_template.  -->

<!-- TODO: @burtenshaw -->

In this section we will explore how Large Language Models (LLMs) structure their generations through tokenization, special tokens, and chat-templates.

## Tokenization

Tokenization is the process of breaking down a text into smaller units, called tokens. These tokens are then used to represent the text in a way that can be processed by the model. For a detailed explanation, see the [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/en/chapter2/4).

For natural language, tokens are frequently appearing combinations of characters within a language. 

Tokenizers are crucial in preparing inputs for models. They convert text into a format that models can understand, typically by splitting text into sub-word units and converting these into numerical IDs. Hugging Face provides two types of tokenizers: a full Python implementation and a "Fast" implementation based on the Rust library 🤗 Tokenizers. The "Fast" tokenizers offer significant speed improvements and additional methods for mapping between the original text and token space.

### Types of Tokenization

1. **Word Tokenization**: This involves splitting text into individual words. It's simple but can be inefficient for languages with complex morphology.

2. **Subword Tokenization**: This breaks down words into smaller units, such as prefixes, suffixes, or even individual characters. This method is more efficient for handling rare words and is commonly used in modern NLP models.

3. **Character Tokenization**: This splits text into individual characters. While it can handle any text, it often results in longer sequences, which can be computationally expensive.

### Importance of Tokenization

- **Efficiency**: Tokenization reduces the complexity of text data, making it easier for models to process.
- **Handling Rare Words**: Subword tokenization helps in managing rare or unseen words by breaking them into known subword units.
- **Language Agnostic**: Tokenization can be adapted to different languages, making it versatile for multilingual models.

## Special Tokens

Special tokens are different for every model and allow segmentation of generation in messages. Can go from messages to prompt with the chat_template. 

## Chat-Templates

Chat templates are essential for structuring interactions between language models and users. They provide a consistent format for conversations, ensuring that models understand the context and role of each message while maintaining appropriate response patterns.

### Base Models vs Instruct Models

A base model is trained on raw text data to predict the next token, while an instruct model is fine-tuned specifically to follow instructions and engage in conversations. For example, `SmolLM2-135M` is a base model, while `SmolLM2-135M-Instruct` is its instruction-tuned variant.

To make a base model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in. ChatML is one such template format that structures conversations with clear role indicators (system, user, assistant).

It's important to note that a base model could be fine-tuned on different chat templates, so when we're using an instruct model we need to make sure we're using the correct chat template.

### Understanding Chat Templates

At their core, chat templates define how conversations should be formatted when communicating with a language model. They include system-level instructions, user messages, and assistant responses in a structured format that the model can understand. This structure helps maintain consistency across interactions and ensures the model responds appropriately to different types of inputs. Below is an example of a chat template:

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

The `transformers` library will take care of chat templates for you in relation to the model's tokenizer. Read more about how transformers builds chat templates [here](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates). All we have to do is structure our messages in the correct way and the tokenizer will take care of the rest. Here's a basic example of a conversation:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

Let's break down the above example, and see how it maps to the chat template format.

### System Messages

System messages set the foundation for how the model should behave. They act as persistent instructions that influence all subsequent interactions. For example:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

### Conversations

Chat templates maintain context through conversation history, storing previous exchanges between users and the assistant. This allows for more coherent multi-turn conversations:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

Templates can handle complex multi-turn conversations while maintaining context:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

### Custom Formatting

You can customize how different message types are formatted. For example, adding special tokens or formatting for different roles:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

### Tool Usage

Tool usage in chat templates allows models to interact with external functions and APIs in a structured way. AI agents rely on tools to perform tasks, such as searching the web, performing calculations, or even controlling physical robots. 

When working with tools, chat templates need to handle three specific message types: tool definitions that describe available functions, tool calls that occur when the assistant wants to use a tool, and tool responses that contain results returned from tool execution.

Here's an example of how a tool interaction might look in a chat template:

```sh
<|im_start|>system
You are an AI assistant with access to a calculator tool.<|im_end|>
<|im_start|>user
What is 123 multiplied by 456?<|im_end|>
<|im_start|>tool
Tool Name: calculator
Tool Arguments: {"operation": "multiply", "x": 123, "y": 456}<|im_end|>
<|im_start|>assistant
Based on the calculator tool, 123 multiplied by 456 equals 56,088.<|im_end|>
```

This example shows how special tokens (`<|im_start|>` and `<|im_end|>`) are used to segment different parts of the conversation, including system context, user input, tool usage, and the assistant's response. Let's see it in action with an example conversation:

```python
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "tool", "tool_name": "WeatherAPI", "args": {"location": "Paris"}},
    {"role": "assistant", "content": "It's currently 20°C and partly cloudy."},
]
```

To use this template with your model, you'll need to ensure your model is trained to work with tool calls in this format. Then, configure your tokenizer with the appropriate chat template. Finally, use the template to format messages before sending to the model:

```python
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
```

Remember that different models may expect different formatting for tool interactions. Always check your model's documentation for the specific format it expects. The template shown here uses a common format with `<|im_start|>` and `<|im_end|>` tokens, but your model might use different special tokens or formatting.

## Resources

- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates) 

