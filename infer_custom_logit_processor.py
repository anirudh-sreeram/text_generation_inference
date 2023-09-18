import sys
import glob, pandas as pd, numpy as np, re
import transformers
from transformers import pipeline
from tqdm import tqdm
from datasets import load_dataset,concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from custom_logits_processor import (NoRepeatNGramLogitsProcessor,)
import torch

def model_init(model_id, cache):
    # creating model and tokenizer
    model_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache,
        trust_remote_code=True,
        use_cache=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model_base, tokenizer

def infer(tokenizer, model_base, prompt,data_path):
    collect_output = []
    for inputs in tqdm(data_path):
        custom_logits_processors = LogitsProcessorList()
        no_repeat_ngram_size = 10
        custom_logits_processors.append(
            NoRepeatNGramLogitsProcessor(no_repeat_ngram_size, tokenizer)
        )

        inputs = inputs.split('<|end|>\n<|user|>\n')[0] + '<|end|>\n<|user|>\n' + prompt + '<|end|>\n<|assistant|>'
            
        cuda_device = 'cuda:0'
        inputs_tokenized = tokenizer(inputs, padding=True, return_tensors="pt")

        with torch.no_grad():
            inputs_tokenized = {k: v.to(cuda_device) for k, v in inputs_tokenized.items()}
            outputs = model_base.generate(
                    input_ids=inputs_tokenized["input_ids"],
                    attention_mask=inputs_tokenized["attention_mask"],
                    max_new_tokens=500,
                    temperature=0.3,
                    num_beams=1,
                    use_cache=True,
                    do_sample=True,
                    logits_processor=custom_logits_processors,
                    num_return_sequences=1,
                    repetition_penalty=1.05
            )

            outputs = outputs[:, inputs_tokenized["input_ids"].shape[1] :]

            single_result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )
            collect_output.append(single_result)
    return collect_output


if __name__ == "__main__":
    model_id = '/mnt/atg_platform/models/now_llm_chat/v0.4.0-rc2'
    cache = 'cache_model'
    data_path = ""
    prompt = ""
    model_base, tokenizer = model_init(model_id, cache)
    output = infer(tokenizer, model_base, prompt, data_path)
    print(output)
