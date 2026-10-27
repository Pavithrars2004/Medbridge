"""Model loading and batched greedy generation (simplification and translation)."""
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C


def _batches(items, size):
    """Yield index batches sorted by text length so padding is small."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]))
    for s in range(0, len(order), size):
        yield order[s:s + size]


class Simplifier:
    """Qwen2.5-3B-Instruct: rewrites a clinical snippet in plain English."""

    def __init__(self, model_name=C.LLM_MODEL):
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda").eval()

    def __call__(self, texts):
        out = [None] * len(texts)
        for idx in _batches(texts, C.SIMPLIFY_BATCH):
            prompts = []
            for i in idx:
                msgs = [{"role": "system", "content": C.SYSTEM_PROMPT},
                        {"role": "user", "content": f"Clinical note:\n{texts[i]}\n\nSimplified explanation:"}]
                prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=C.MAX_NEW_TOKENS, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            gen = gen[:, enc["input_ids"].shape[1]:]
            for i, g in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = g.strip()
        return out


class Translator:
    """NLLB-200 distilled 600M: translates between English and the six Indian languages."""

    def __init__(self, model_name=C.MT_MODEL):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()

    def __call__(self, texts, src_code, tgt_code, repetition_penalty=1.0):
        out = [None] * len(texts)
        self.tok.src_lang = src_code
        forced = self.tok.convert_tokens_to_ids(tgt_code)
        for idx in _batches(texts, C.TRANSLATE_BATCH):
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=C.MAX_LENGTH).to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=forced, max_length=C.MAX_LENGTH,
                                          num_beams=1, do_sample=False, repetition_penalty=repetition_penalty)
            for i, g in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = g.strip()
        return out
