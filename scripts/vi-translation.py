import os
import sys
import re
from huggingface_hub import InferenceClient

PROMPT = lambda content: f'''
You are a translator for the Vietnamese translation team. You are tasked with translating the following text into Vietnamese. You must follow these instructions:
- Translate the text into Vietnamese, while keeping the original formatting (either Markdown, MDX or HTML)
- Inside code blocks, translate the comments but leave the code as-is ; If the code block contains quite plain texts, you MUST provide the translation in <details> tag.
- Do not translate inline code, the URLs and file paths
- If the term is abbreviated, keep the original term and provide the translation in parentheses for the first time it appears in the text.
- If there are any slag or funny joke in english, keep it (do not translate) and give an explanation so vietnamese reader can understand.
- Use "ta", "chúng ta", "chúng mình", "các bạn" as pronouns.

KEEP THESE TERMS (DO NOT TRANSLATE, do NOT add translation in parentheses): model, API, SDK, CLI, HTML, GGUF, AI, training, inference, server, client, notebook, python, Hugging Face, transformers, diffusion, diffuser, data, function, LangGraph, LangChain, Llama, Gemma, token, Unit, pretrain, Live (live stream), form, format, certificate, Space, CodeAgent

Also KEEP these terms but PROVIDE TRANSLATION in parentheses for the first time it appears in the text: alignment (cân chỉnh), LLM, RAG (Tìm kiếm và tạo ra câu trả lời), Agent (tác nhân), Tools (công cụ), "Special Token" (Token đặc biệt), "chain-of-thought" (luồng suy luận), fine-tuning (tinh chỉnh), Thought-Action-Observation

For these terms, use the pre-defined translation:
- Quick Quiz: Kiểm tra nhanh
- Unit: Chương
- Bonus Unit: Chương bổ trợ
- Module: Mô-đun
- Lesson ...: Bài ...
- Course: Khóa học
- state-of-the-art: nổi tiếng
- Q&A: Hỏi và Đáp
- Dummy: ảo (or "giả", or "thử" depending on the context)
- onboarding: làm quen
- Hands-on: Thực hành
- Challenge: Bài tập lớn

Here is an example:
- Original text: To run the models, we will use [ollama](https://ollama.com), a command line tool that allows you to run LLMs and embedding models from Hugging Face. With ollama, you **don't need** to have access to a server or cloud service to run the models. You can run the models directly **on your computer**.
- Translation: Để chạy các model, ta sẽ sử dụng [ollama](https://ollama.com), một công cụ dòng lệnh cho phép bạn chạy LLMs và embedding models từ Hugging Face. Với ollama, bạn **không cần** phải tạo server hay truy cập API bên thứ 3. Bạn có thể chạy các model trực tiếp **trên máy tính của bạn**.

Here is another example:
- Original text: The model can then be **aligned** to the creator's preferences. For instance, a customer service chat model that must never be impolite to customers.
- Translation: Model sau đó có thể được **alignment** (cân chỉnh) theo mong muốn của người tạo. Ví dụ: model chat hỗ trợ khách hàng không bao giờ được bất lịch sự.

If the code block contains many plain texts, prove translation in collapsible <details> tag. Example:
- Original text:
    ```
    <|im_start|>Hello, how are you?<|im_end|>
    <|im_start|>I'm fine, thank you.<|im_end|>
    message = {{"user": "This is a test"}}
    ```
- Translation (add the <details> collapsible ABOVE of the original code block):
    <details>
    <summary>Bấm để xem bản dịch tiếng Việt</summary>
    ```
    <|im_start|>Xin chào, bạn có khỏe không?<|im_end|>
    <|im_start|>Mình khỏe, cảm ơn bạn.<|im_end|>
    message = {{"user": "Đây là một tin nhắn thử"}}
    ```
    </details>
    ```
    <|im_start|>Hello, how are you?<|im_end|>
    <|im_start|>I'm fine, thank you.<|im_end|>
    message = {{"user": "This is a test"}}
    ```


IMPORTANT: Only output the translated text and nothing else, no need explaination or instruction. The input text is between "=== BEGIN OF TEXT ===" and "=== END OF TEXT ===".

Please translate the following text to vietnamese:

=== BEGIN OF TEXT ===
{content}
=== END OF TEXT ===
'''.strip()

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
inp_dir = os.path.join(script_dir, '..', 'units/en')
get_our_path = lambda x: x.replace('/en', '/vi')
model = "deepseek-ai/DeepSeek-R1"
client = InferenceClient(
	provider="together",
    # api_key is read from the environment
)

def auto_translate(
    inp_dir: str,
    get_our_path: callable,
    model: str,
    client: InferenceClient,
    PROMPT: callable
):
    escape_special_tokens = lambda x: x.replace('<think>', '<%%think%%>').replace('</think>', '<%%/think%%>')
    unescape_special_tokens = lambda x: x.replace('<%%think%%>', '<think>').replace('<%%/think%%>', '</think>')

    # Get the list of all files in the directory, recursively
    inp_files: list[str] = []
    print('Collecting files...')
    for root, dirs, files in os.walk(inp_dir):
        for file in files:
            if file.endswith('.mdx') or file == "_toctree.yml":
                fname = os.path.join(root, file)
                print('  +', fname)
                inp_files.append(fname)

    def write_out_file(fpath: str, content: str):
        base_path = os.path.dirname(fpath)
        os.makedirs(base_path, exist_ok=True)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)

    # Read the content of the file and process
    for i, inp_file in enumerate(inp_files):
        out_file = get_our_path(inp_file)
        if os.path.exists(out_file):
            print(f'[{i+1}/{len(inp_files)}] Skipping file: {inp_file}')
            continue
        with open(inp_file, 'r', encoding='utf-8') as f:
            content: str = f.read()
            content = escape_special_tokens(content)
            if content.strip() == "":
                print(f'[{i+1}/{len(inp_files)}] Skipping empty file: {inp_file}')
                write_out_file(out_file, "")
                continue

            print(f'[{i+1}/{len(inp_files)}] Processing file: {inp_file}')
            stream = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": PROMPT(content)},
                ],
                stream=True,
            )
            final_text = ""
            for chunk in stream:
                print(chunk.choices[0].delta.content, end="")
                sys.stdout.flush()
                final_text += chunk.choices[0].delta.content
            # Optionally filter <think>...</think> reasoning process
            final_text = final_text.split('</think>').pop().strip()
            # Write the output to the file
            final_text = unescape_special_tokens(final_text)
            write_out_file(out_file, final_text)
            print()
            print(f'  -> Translated to: {out_file}')
            print("--" * 20)
            #break

if __name__ == '__main__':
    auto_translate(inp_dir, get_our_path, model, client, PROMPT)
