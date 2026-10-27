# %% [markdown]
# # MedBridge: Faithfulness Study (Full Colab Pipeline)
# Qwen2.5-3B-Instruct (simplify) + NLLB-200 distilled 600M (translate), 6 Indian languages, 2 pipeline orders, scoring, statistics.
# **Before you start:** Runtime > Change runtime type > T4 GPU > Save. Then run the cells from top to bottom.

# %% [markdown]
# ## Cell 1: Install, GPU check and upload the evaluation set

# %%
import torch
assert torch.cuda.is_available(), "No GPU. Go to Runtime > Change runtime type > T4 GPU, Save, then Runtime > Restart session and rerun."
print("GPU:", torch.cuda.get_device_name(0))

!pip install -q transformers accelerate sentencepiece pandas numpy

import os, re, math, random, time, zipfile
import numpy as np
import pandas as pd
from google.colab import files

print("Upload final_evaluation_set.csv (needs columns SNIPPET_ID and SNIPPET)")
if not os.path.exists("final_evaluation_set.csv"):
    files.upload()
df = pd.read_csv("final_evaluation_set.csv")
assert {"SNIPPET_ID", "SNIPPET"}.issubset(df.columns), "CSV must have SNIPPET_ID and SNIPPET columns"
df = df[["SNIPPET_ID", "SNIPPET"]].dropna().reset_index(drop=True)
print("Loaded", len(df), "snippets")

# %% [markdown]
# ## Cell 2: Settings (same as the report: greedy decoding, batch 8 and 16, fixed seed)

# %%
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MT_MODEL = "facebook/nllb-200-distilled-600M"

LANG_CODES = {
    "Telugu": "tel_Telu", "Tamil": "tam_Taml", "Kannada": "kan_Knda",
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Marathi": "mar_Deva",
}
SIMPLIFY_BATCH = 8
TRANSLATE_BATCH = 16
MAX_NEW_TOKENS = 300
N_BOOT = 5000
OUT_DIR = "medbridge_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## Cell 3: Load the two models and define batched simplify / translate (greedy decoding)

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL, padding_side="left")
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float16, device_map="cuda")
llm.eval()

mt_tok = AutoTokenizer.from_pretrained(MT_MODEL)
mt = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, torch_dtype=torch.float16).to("cuda")
mt.eval()

SYSTEM_PROMPT = ("You rewrite clinical notes in simple, plain English for a patient. "
                 "Do not add any new medical facts. Preserve every number, medication name, dose, and negation exactly.")

def _batches(items, size):
    # sort by length so each batch has similar lengths, then restore the order
    order = sorted(range(len(items)), key=lambda i: len(items[i]))
    for s in range(0, len(order), size):
        yield order[s:s + size]

def simplify_batch(texts):
    out = [None] * len(texts)
    for idx in _batches(texts, SIMPLIFY_BATCH):
        prompts = []
        for i in idx:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Clinical note:\n{texts[i]}\n\nSimplified explanation:"}]
            prompts.append(llm_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = llm_tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = llm.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                               pad_token_id=llm_tok.eos_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        for i, g in zip(idx, llm_tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = g.strip()
    return out

def translate_batch(texts, src_code, tgt_code, repetition_penalty=1.0):
    out = [None] * len(texts)
    mt_tok.src_lang = src_code
    forced = mt_tok.convert_tokens_to_ids(tgt_code)
    for idx in _batches(texts, TRANSLATE_BATCH):
        enc = mt_tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                     truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            gen = mt.generate(**enc, forced_bos_token_id=forced, max_length=512, num_beams=1,
                              do_sample=False, repetition_penalty=repetition_penalty)
        for i, g in zip(idx, mt_tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = g.strip()
    return out

# quick dry run on 3 snippets
test = df["SNIPPET"].head(3).tolist()
simp = simplify_batch(test)
print(simp[0][:300]); print("-" * 40)
print(translate_batch(simp[:1], "eng_Latn", LANG_CODES["Telugu"])[0][:300])

# %% [markdown]
# ## Cell 4: Run both pipeline orders for all snippets and all 6 languages (about 40 to 50 minutes on a T4)
# - **Translate-only:** English snippet goes straight into NLLB-200.
# - **Simplify-then-translate:** Qwen simplifies in English first, then NLLB-200 translates.
# The result is saved to disk after each language, so a disconnect does not lose everything.

# %%
t0 = time.time()
snips = df["SNIPPET"].tolist()
ids = df["SNIPPET_ID"].tolist()

simplified = simplify_batch(snips)
pd.DataFrame({"SNIPPET_ID": ids, "SIMPLIFIED_TEXT": simplified}).to_csv(f"{OUT_DIR}/simplified_texts.csv", index=False)
print(f"Simplification done in {(time.time()-t0)/60:.1f} min")

rows = []
for lang, code in LANG_CODES.items():
    a = translate_batch(snips, "eng_Latn", code)         # translate-only
    b = translate_batch(simplified, "eng_Latn", code)    # simplify then translate
    for sid, o in zip(ids, a):
        rows.append({"SNIPPET_ID": sid, "LANGUAGE": lang, "ORDER": "translate_only_baseline", "OUTPUT": o})
    for sid, o in zip(ids, b):
        rows.append({"SNIPPET_ID": sid, "LANGUAGE": lang, "ORDER": "simplify_then_translate", "OUTPUT": o})
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/pipeline_outputs.csv", index=False)
    print(f"{lang} done | total {(time.time()-t0)/60:.1f} min")

outputs = pd.DataFrame(rows)
print(len(outputs), "outputs")   # expected 250 x 6 x 2 = 3000

# %% [markdown]
# ## Cell 5: Back-translate every output to English (needed for the negation and abnormal-flag check)

# %%
def back_translate(outputs_df, repetition_penalty=1.0, languages=None):
    res = outputs_df.copy()
    res["BACKTRANSLATION"] = ""
    for lang, code in LANG_CODES.items():
        if languages and lang not in languages:
            continue
        m = res["LANGUAGE"] == lang
        res.loc[m, "BACKTRANSLATION"] = translate_batch(res.loc[m, "OUTPUT"].tolist(), code, "eng_Latn",
                                                        repetition_penalty=repetition_penalty)
        print("back-translated", lang)
    return res

bt = back_translate(outputs)
bt.to_csv(f"{OUT_DIR}/pipeline_outputs_backtranslated.csv", index=False)

# %% [markdown]
# ## Cell 6: Ground truth answer key (extracted automatically from each source snippet)
# Four signal types: negations, medication doses, lab values, abnormal flags.

# %%
PLACEHOLDER = re.compile(r"\[\*\*.*?\*\*\]")       # de-identification tags such as [**2142-6-9**]
NUM = re.compile(r"\d+(?:\.\d+)?")

NEG_PATTERNS = [r"no evidence of", r"denies", r"denied", r"without", r"negative for", r"absence of",
                r"\bno\b", r"\bnot\b", r"\bnever\b"]
FLAG_WORDS = ["elevated", "decreased", "abnormal", "critical", "high", "low", "increased", "reduced"]
DOSE_RE = re.compile(r"\b([A-Za-z][A-Za-z\-]{2,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|units?|ml|meq|iu)\b", re.I)
LAB_RE = re.compile(r"\b(INR|PTT|PT|WBC|Hct|Hgb|Hb|plts?|platelets?|Cr|creatinine|BUN|K|Na|glucose|Tbili|bilirubin|"
                    r"lactate|troponin|ferritin|fibrinogen|albumin|ALT|AST|ALP|CK)\s*[:=]?\s*(\d+(?:\.\d+)?)", re.I)

def clean(text):
    return PLACEHOLDER.sub(" ", str(text))

def extract_ground_truth(text):
    t = clean(text)
    low = t.lower()
    negs = sorted({p.replace(r"\b", "") for p in NEG_PATTERNS if re.search(p, low)})
    doses = [f"{m.group(1)} {m.group(2)} {m.group(3)}".lower() for m in DOSE_RE.finditer(t)]
    labs = [f"{m.group(1)} {m.group(2)}".lower() for m in LAB_RE.finditer(t)]
    flags = [w for w in FLAG_WORDS if re.search(r"\b" + w + r"\b", low)]
    return {"numbers": sorted(set(NUM.findall(t)), key=float),
            "negations": negs, "doses": doses, "labvalues": labs, "flags": flags}

gt = {sid: extract_ground_truth(txt) for sid, txt in zip(df["SNIPPET_ID"], df["SNIPPET"])}
key = pd.DataFrame([{"SNIPPET_ID": k, "GT_NUMBERS": "|".join(v["numbers"]), "GT_NEGATIONS": "|".join(v["negations"]),
                     "GT_DOSES": "|".join(v["doses"]), "GT_LABVALUES": "|".join(v["labvalues"]),
                     "GT_ABNORMAL_FLAGS": "|".join(v["flags"])} for k, v in gt.items()])
key.to_csv(f"{OUT_DIR}/ground_truth_answer_key.csv", index=False)
print(key.head())
print("snippets with a negation cue:", (key.GT_NEGATIONS != "").sum(), "| with an abnormal flag:", (key.GT_ABNORMAL_FLAGS != "").sum())

# %% [markdown]
# ## Cell 7: Scoring (numeric fidelity, hallucinated numbers, negation and flag survival)

# %%
def _digits(base):
    return {base + i: ord("0") + i for i in range(10)}
DIGIT_MAP = {}
for base in (0x0966, 0x09E6, 0x0BE6, 0x0C66, 0x0CE6):   # Devanagari, Bengali, Tamil, Telugu, Kannada
    DIGIT_MAP.update(_digits(base))

def normalize_digits(text):
    return str(text).translate(DIGIT_MAP)

def numbers_in(text):
    return set(NUM.findall(normalize_digits(clean(text))))

def numeric_scores(source_text, output_text):
    src = numbers_in(source_text)
    out = numbers_in(output_text)
    if not src:
        return None, None, False
    preserved = src & out
    halluc = out - src
    return len(preserved) / len(src) * 100, sorted(halluc), len(halluc) > 0

def cue_survives(back_text, cues):
    low = str(back_text).lower()
    return any(c in low for c in cues)

NEG_CUES_EN = ["no evidence of", "denies", "denied", "without", "negative for", "absence of", "no ", "not ", "never", "none"]
FLAG_CUES_EN = ["elevated", "decreased", "abnormal", "critical", "high", "low", "increased", "reduced", "raised", "lowered"]

def score_all(bt_df):
    src = dict(zip(df["SNIPPET_ID"], df["SNIPPET"]))
    rec = []
    for r in bt_df.itertuples():
        g = gt[r.SNIPPET_ID]
        fid, halluc, has_h = numeric_scores(src[r.SNIPPET_ID], r.OUTPUT)
        neg_ok = cue_survives(r.BACKTRANSLATION, NEG_CUES_EN) if g["negations"] else None
        flag_ok = cue_survives(r.BACKTRANSLATION, FLAG_CUES_EN) if g["flags"] else None
        rec.append({"SNIPPET_ID": r.SNIPPET_ID, "LANGUAGE": r.LANGUAGE, "ORDER": r.ORDER,
                    "NUMERIC_FIDELITY_PCT": fid, "HALLUCINATED_NUMBERS": "|".join(halluc or []),
                    "HAS_HALLUCINATED_NUMBER": has_h, "NEGATION_PRESERVED": neg_ok, "FLAG_PRESERVED": flag_ok})
    return pd.DataFrame(rec)

scored = score_all(bt)
scored.to_csv(f"{OUT_DIR}/scored_outputs.csv", index=False)

summary = (scored.groupby(["LANGUAGE", "ORDER"])
           .agg(n=("SNIPPET_ID", "count"),
                numeric_fidelity=("NUMERIC_FIDELITY_PCT", "mean"),
                hallucinated_number_rate=("HAS_HALLUCINATED_NUMBER", lambda s: s.mean() * 100),
                negation_survival=("NEGATION_PRESERVED", lambda s: s.dropna().astype(float).mean() * 100),
                flag_survival=("FLAG_PRESERVED", lambda s: s.dropna().astype(float).mean() * 100))
           .round(1).reset_index())
summary.to_csv(f"{OUT_DIR}/results_summary.csv", index=False)
summary

# %% [markdown]
# ## Cell 8: Statistics (paired sign test and 95% bootstrap confidence interval, plain Python)

# %%
def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return None
    z = (pos - n / 2) / math.sqrt(n / 4)
    return 2 * (1 - normal_cdf(abs(z)))

def bootstrap_ci(diffs, n_resamples=N_BOOT, ci=0.95, seed=SEED):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int((1 - ci) / 2 * n_resamples)]
    hi = means[int((1 + ci) / 2 * n_resamples) - 1]
    return lo, hi

def paired_diffs(col, lang):
    sub = scored[scored.LANGUAGE == lang]
    a = sub[sub.ORDER == "translate_only_baseline"].set_index("SNIPPET_ID")[col]
    b = sub[sub.ORDER == "simplify_then_translate"].set_index("SNIPPET_ID")[col]
    j = pd.concat([a, b], axis=1, keys=["t", "s"]).dropna().astype(float)
    return (j["t"] - j["s"]).tolist()      # positive = translate-only is better

rows = []
for lang in LANG_CODES:
    for col, name in [("NUMERIC_FIDELITY_PCT", "numeric_fidelity"), ("NEGATION_PRESERVED", "negation_survival"),
                      ("FLAG_PRESERVED", "flag_survival")]:
        d = paired_diffs(col, lang)
        if not d:
            continue
        scale = 1 if col == "NUMERIC_FIDELITY_PCT" else 100
        lo, hi = bootstrap_ci(d)
        rows.append({"LANGUAGE": lang, "METRIC": name, "N": len(d),
                     "MEAN_GAP_POINTS": round(sum(d) / len(d) * scale, 1),
                     "CI95_LOW": round(lo * scale, 1), "CI95_HIGH": round(hi * scale, 1),
                     "SIGN_TEST_P": sign_test(d)})
stats = pd.DataFrame(rows)
stats.to_csv(f"{OUT_DIR}/statistical_significance.csv", index=False)
stats

# %% [markdown]
# ## Cell 9 (optional ablation): Telugu back-translation with repetition penalty 1.3
# This is the decoding fix described in Section 6.6 (it stops repeated-phrase loops that hide a negation).

# %%
tel = outputs[outputs.LANGUAGE == "Telugu"].copy()
bt13 = back_translate(tel, repetition_penalty=1.3, languages=["Telugu"])
s13 = score_all(bt13)
s10 = scored[scored.LANGUAGE == "Telugu"]
abl = pd.DataFrame({
    "setting": ["repetition_penalty 1.0", "repetition_penalty 1.3"],
    "negation_survival": [s10.NEGATION_PRESERVED.dropna().astype(float).mean() * 100,
                          s13.NEGATION_PRESERVED.dropna().astype(float).mean() * 100],
    "flag_survival": [s10.FLAG_PRESERVED.dropna().astype(float).mean() * 100,
                      s13.FLAG_PRESERVED.dropna().astype(float).mean() * 100]}).round(1)
abl.to_csv(f"{OUT_DIR}/ablation_repetition_penalty_telugu.csv", index=False)
abl

# %% [markdown]
# ## Cell 10: Download everything as one zip

# %%
zip_name = "medbridge_results.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(OUT_DIR):
        z.write(os.path.join(OUT_DIR, f), f)
files.download(zip_name)
