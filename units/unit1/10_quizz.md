# Unit 1 Quiz: Fundamentals of AI Agents


## Quiz on Large Language Models (LLMs)

1. Which of the following best describes a Large Language Model (LLM)?
   - A. A model specializing in language recognition  
   - B. A massive neural network that understands and generates human language  
   - C. A model exclusively used for language data tasks like summarization or classification  
   - D. A rule-based chatbot used for conversations  

2. LLMs are typically:  
   - A. Pre-trained on small, curated datasets  
   - B. Trained on large text corpora to capture linguistic patterns  
   - C. Trained purely on translation tasks  
   - D. Designed to function solely with GPU resources  

3. Which of the following is a common architecture for LLMs?  
   - A. Convolutional Neural Networks (CNNs)  
   - B. Transformer  
   - C. Recurrent Neural Networks (RNNs) with LSTM  
   - D. Support Vector Machines  

4. What does it mean when we say LLMs are "autoregressive"?  
   - A. They regress to the mean to reduce variance  
   - B. They generate text by predicting the next token based on previous tokens  
   - C. They can only handle labeled data  
   - D. They can output text only after the entire input is known at once  

5. Which of these is NOT a common use of LLMs?  
   - A. Summarizing content  
   - B. Generating code  
   - C. Playing strategy games like chess or Go  
   - D. Conversational AI  

## Quiz on Messages and Special Tokens

1. Which of the following best describes a "special token"?
   - A. A token that makes the model forget all context  
   - B. A model signature required for API calls  
   - C. A token that helps segment or structure the conversation in the model  
   - D. A token that always represents the end of text  

2. What is the primary goal of a "chat template"?  
   - A. To force the model into a single-turn conversation  
   - B. To structure interactions and define roles in a conversation  
   - C. To replace the need for system messages  
   - D. To store prompts into the model's weights permanently  

3. How do tokenizers handle text for modern NLP models?  
   - A. By splitting text into individual words only  
   - B. By breaking words into subword units and assigning numerical IDs  
   - C. By storing text directly without transformation  
   - D. By removing all punctuation automatically  

4. Which role in a conversation sets the overall behavior for a model?
   - A. user  
   - B. system  
   - C. assistant  
   - D. developer  

5. Which statement is TRUE about tool usage in chat templates?  
   - A. Tools cannot be used within the conversation context.  
   - B. Tools are used only for logging messages.  
   - C. Tools allow the assistant to offload tasks like web search or calculations.  
   - D. Tools are unsupported in all modern LLMs.  


<!-- TODO: @burtenshaw add more questions on completion -->

<details>
<summary>solution</summary>

### LLMs Solutions (Updated)
1. **Question**: Which of the following best describes a Large Language Model (LLM)?  
   **Answer**: B. A massive neural network that understands and generates human language  
   **Explanation**: LLMs have millions to billions of parameters and can capture complex patterns in large text corpora.

2. **Question**: LLMs are typically:  
   **Answer**: B. Trained on large text corpora to capture linguistic patterns  
   **Explanation**: Training on large datasets helps them learn grammar, context, and various linguistic structures.

3. **Question**: Which of the following is a common architecture for LLMs?  
   **Answer**: B. Transformer  
   **Explanation**: The Transformer architecture is widely used by modern LLMs like Llama, GPT and BERT.

4. **Question**: What does it mean when we say LLMs are "autoregressive"?  
   **Answer**: B. They generate text by predicting the next token based on previous tokens  
   **Explanation**: Autoregressive generation means output tokens are predicted sequentially, one after another.

5. **Question**: Which of these is NOT a common use of LLMs?  
   **Answer**: C. Playing strategy games like chess or Go  
   **Explanation**: While LLMs excel at language tasks, they are not typically used for strategy board games.

### Messages and Special Tokens Solutions (Unchanged)
1. **Question**: Which of the following best describes a "special token"?  
   **Answer**: C. A token that helps segment or structure the conversation in the model  
   **Explanation**: Special tokens often indicate the start/end of roles or separate different parts of the conversation.

2. **Question**: What is the primary goal of a "chat template"?  
   **Answer**: B. To structure interactions and define roles in a conversation  
   **Explanation**: Chat templates maintain format consistency, ensuring the model knows who is speaking and why.

3. **Question**: How do tokenizers handle text for modern NLP models?  
   **Answer**: B. By breaking words into subword units and assigning numerical IDs  
   **Explanation**: Subword tokenization lets models handle rare or unknown words by splitting them into smaller chunks.

4. **Question**: Which role in a conversation sets the overall behavior for a model?  
   **Answer**: B. system  
   **Explanation**: System messages set global instructions that the model follows during the entire conversation.

5. **Question**: Which statement is TRUE about tool usage in chat templates?  
   **Answer**: C. Tools allow the assistant to offload tasks like web search or calculations.  
   **Explanation**: Tools extend model functionality by allowing external function calls within a conversation context.

</details>
