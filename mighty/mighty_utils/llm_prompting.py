"""Contains prompting utilities for LLMs: currently Llama & Qwen via HF."""

"""The code here is based on the official examples."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_llm(model_name):
    """Get the LLM model and tokenizer."""
    if "qwen" in model_name:
        return make_qwen(model_name), prompt_qwen
    elif "llama" in model_name:
        return make_llama(model_name), prompt_llama
    else:
        raise ValueError(
            f"Model {model_name} not supported. Currently only Qwen and Llama are supported."
        )


def make_qwen(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer


def prompt_qwen(mt, prompt, add_generation_prompt=True, enable_thinking=True):
    model, tokenizer = mt
    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    return tokenizer.decode(output_ids, skip_special_tokens=True)


def make_llama(model_name):
    return pipeline(model=model_name, device=device, torch_dtype=torch.bfloat16)


def prompt_llama(
    model,
    prompt,
    system_prompt="You are a helpful assistant, that responds as a pirate.",
):
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    generation = model(
        prompt, do_sample=False, temperature=1.0, top_p=1, max_new_tokens=50
    )
    return generation[0]["generated_text"]
