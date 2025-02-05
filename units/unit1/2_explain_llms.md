# Explain Large Language Models

<!-- Explanation of LLMs, including the family tree of models: encoders, seq2seq, decoders. Decoders are autoregressive and continue until EOS.  -->

<!-- TODO: @burtenshaw -->
As explained in the previous chapter, each agent needs an AI Model at his core. And the currently most used AI model for those are LLMs ( Large Language Model).

## What is a Large Language Model?
A Large Language Model (LLM) is a type of artificial intelligence model that excels at understanding and generating human language. They are trained on vast amounts of text data, allowing them to learn patterns, nuances, and structure in language. These models typically consist of billions of parameters

Most LLMs are **Transformers**, it is an architecture of deep learning models that has gained a lot of interest since the release of Bert from google in 2018. There are 3 types of transformers :

1. **Encoders**  
   An encoder-based Transformer takes text (or other data) as input and outputs a dense representation (or embedding) of that text.

   - **Example**: BERT from Google
   - **Use Cases**: Text classification, semantic search, Named Entity Recognition
   - **Usual Size**: Millions of parameters

2. **Decoders**  
   A decoder-based Transformer is focused on _generating_ new token to complete a sequence token-by-token.

   - **Example**: LLama from meta 
   - **Use Cases**: Text generation, chatbots, code generation
   - **Usual Size**: Billions of parameters

3. **Seq2Seq (Encoder–Decoder)**  
   A sequence-to-sequence Transformer _combines_ an encoder and a decoder. The encoder first processes the input sequence into a context representation, then the decoder generates an output sequence.

   - **Example**: T5, BART, 
   - **Use Cases**:  Translation, Summarization, Paraphrasing
   - **Usual Size**: Millions of parameters

Although Large Language Models come in various forms, in most people's mind LLMs are **decoders** of multiple billions parameterss. Here are some of it's famous names :

| **Model**                          | **Provider**                              |
|-----------------------------------|-------------------------------------------|
| **GPT4**                           | OpenAI                                    |
| **LLaMA3**                         | Meta (Facebook AI Research)               |
| **Deepseek-R1**                    | DeepSeek                                  |
| **SmollLM2**                          | BigScience (collaborative initiative)     |
| **Gemma**                          | Google                                    |
| **Mistral**                        | Mistral                                |

The principle behind LLM is very simple yet very effective. The objective of a decoder is to **Predict the next token**. We talk about tokens and not words because not every token translates to a word. In English, the dictionarry have an estimated 600 000 different words while the vocabulary size of an LLM is around 32 000 tokens ( for Llama2 ). This can be achieved by functioning on Sub-word token.

For example you can see interesting as "Interest" + "##ing" which can both be reused to compose new word like "interested" ( "interest" + "##ed" ) or "fasting" ( "fast" + "##ing")

You can play with different tokenizers in the space below:

<iframe
	src="https://xenova-the-tokenizer-playground.static.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

Furthermore, each LLM have some **special tokens** specific to this model. The most important of those special token is the **End of sequence token** (EOS).

| **Model**     | **Provider**                  | **EOS Token**                  |
|---------------|-------------------------------|---------------------------------|
| **GPT4**      | OpenAI                        | `<|endoftext|>`                |
| **LLaMA3**    | Meta (Facebook AI Research)   | `<|end_header_id|>`            |
| **Deepseek-R1** | DeepSeek                   | `<｜end▁of▁sentence｜>`         |
| **SmollLM2**  | Hugging Face                 | `<|im_end|>`                   |
| **Gemma**     | Google                       | `<end_of_turn>`                |
| **Mistral**   | Mistral                      | `[/INST]`                      |


## Understanding next token prediction.

LLMs are said to be **autoregrassive**, meaning that the output from one pass become the input from the next one. This loop continue untill the model predict the next token to be the EOS token. At which point the model can stop.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/AutoregressionFinal.gif" alt="Visual Gif of autoregressive decoding" width="60%">

Alright, an LLM will decode untill reaching an EOS. But what happens during a single loop ?

The process is probably a bit too technical for the purpose of learning agents. If you want to know more about decoding, you can take a look at the NLP course.

In short : Once we have tokenized our text, we compute a dense representation of the words that both accounts for their meaning and their position in the input sequence. This Dense representation goes into the model and outputs some logits and those logits can be remapped to a token id ( a unique number for each token ) through the softmax function.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/DecodingFinal.gif" alt="Visual Gif of decoding" width="60%">

Once out of the model, we have multiple options of potential tokens that could complete the sentence. The most naive decoding strategy would be to always take the word with the maximum probability.

You can interract with the decoding process yourself with SmollLM2 in this space (remember, it decodes untill reaching an **EOS** token which is  **<|im_end|>** for this model):

<iframe
	src="https://jofthomas-decoding-visualizer.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

But there is also other more advanced decoding strategies like beam search. Where instead of searching immediate reward, we will to find the maximum cummulative probability by exploring options when taking the sub-optimal options on the shorter term

<iframe
	src="https://m-ric-beam-search-visualizer.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>s

## Attention is all you need
One small detail that we should still mention is **Attention**. When predicting the next word. Not all the words in the sentence have the same importance. For instance, when decoding " The Capital of France is ", the Attention will be higher on the words "France" and "Capital" as they are the ones holding the meaning of the sentence.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/AttentionSceneFinal.gif" alt="Visual Gif of Attention" width="60%">

This simple process of finding the most probable word to complete a sequence proved itself to be very usefull, in fact, the basic principle of LLM did not change much since gpt2, but the sizes of the neural network and the way to change the attention has drastically changed.

If you have interacted with LLMs, you most likely heard about the word "context length". This represents the amount of tokens that the LLM can process as a total and that's what have been the most impacted by the recent improvment in Attention.

IE : Context length is the maximum number of Input token + Output Token that the model can handle.

## Prompting the LLM is important.
Considering that the only job of an LLM is to predict the next token by looking at every input token , and to chose which are "important" to decode what the next word should be, the wording of your input token is very important.

This called a prompt in LLM and will allow to guide the generation of the LLM toward the desired output.

## How do LLMs learn?

LLMs are trained on large datasets of text, where they learn to predict the next word in a sequence through a self-supervised or masked language modeling objective. From this unsupervised learning, the model learns the structure of the language and underlying patterns in text allowing to generalize on unseen data.

Following this, LLMs can be fine-tuned on a supervised learning objective to perform specific tasks. For example, some are trained for conversational structures or tool usage, while others focus on classification or code generation.

## How can I use LLMs?

You can run LLMs locally on your own laptop (if you have sufficient hardware) or call them through an API. Some models require a lot of memory to run efficiently, so you'll need to factor in these hardware requirements when choosing a model.

During this course, we will initially use models through APIs on the Hugging Face hub. Later, we will explore how to run models locally on your own hardware.

## How are LLMs used in AI Agents?

LLMs are a key component of AI Agents, providing the foundation for understanding and generating human language. They can interpret user instructions, maintain context in conversations, and decide which tools to use. We will explore these steps in more detail in [dedicated sections](./6_agent_steps_and_structure.md).


## Challenges and Limitations

While LLMs are powerful, they can sometimes generate biased or factually incorrect content. They may also require substantial computational resources to train and deploy. Researchers are actively exploring methods to reduce these issues and make LLMs more reliable.