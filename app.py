import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

model_name = "RaviNaik/Phi2-Osst"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="cuda:0"
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map="cuda:0")
tokenizer.pad_token = tokenizer.eos_token
chat_template = """<|im_start|>system
You are a helpful assistant who always respond to user queries<|im_end|>
<im_start>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

def generate(prompt, max_length, temperature, num_samples):
    prompt = prompt.strip()
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length, temperature=temperature, num_return_sequences=num_samples)
    # result = pipe(chat_template.format(prompt=prompt))
    result = pipe(prompt)
    return {output: result}


with gr.Blocks() as app:
    gr.Markdown("## ERA Session27 - Phi2 Model Finetuning with QLoRA on OpenAssistant Conversations Dataset (OASST1)")
    gr.Markdown(
        """This is an implementation of [Phi2](https://huggingface.co/microsoft/phi-2) model finetuning using QLoRA stratergy on [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1)
        
        Please find the source code and training details [here](https://github.com/RaviNaik/ERA-SESSION27).
        
        Dataset used to finetune: [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1)
        ChatML modified OSST Dataset: [RaviNaik/oasst1-chatml](https://huggingface.co/datasets/RaviNaik/oasst1-chatml)
        Finetuned Model: [RaviNaik/Phi2-Osst](https://huggingface.co/RaviNaik/Phi2-Osst)
        """
    )
    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(label="Initial Prompt", interactive=True)
            max_length = gr.Slider(
                minimum=50,
                maximum=500,
                value=200,
                step=10,
                label="Select Number of Tokens to be Generated",
                interactive=True,
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1,
                value=0.7,
                step=0.1,
                label="Select Temperature",
                interactive=True,
            )
            num_samples = gr.Dropdown(
                choices=[1, 2, 5, 10],
                value=1,
                interactive=True,
                label="Select No. of outputs to be generated",
            )
            submit_btn = gr.Button(value="Generate")

        with gr.Column():
            output = gr.JSON(label="Generated Text")

        submit_btn.click(
            generate,
            inputs=[prompt_box, max_length, temperature, num_samples],
            outputs=[output],
        )

app.launch()