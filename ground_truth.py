"""Automatic ground-truth answer key: numbers, negations, doses, lab values, abnormal flags."""
import re

import pandas as pd

PLACEHOLDER = re.compile(r"\[\*\*.*?\*\*\]")   # de-identification tags, e.g. [**2142-6-9**]
NUM = re.compile(r"\d+(?:\.\d+)?")

NEG_PATTERNS = [r"no evidence of", r"denies", r"denied", r"without", r"negative for",
                r"absence of", r"\bno\b", r"\bnot\b", r"\bnever\b"]
FLAG_WORDS = ["elevated", "decreased", "abnormal", "critical", "high", "low", "increased", "reduced"]
DOSE_RE = re.compile(r"\b([A-Za-z][A-Za-z\-]{2,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|units?|ml|meq|iu)\b", re.I)
LAB_RE = re.compile(r"\b(INR|PTT|PT|WBC|Hct|Hgb|Hb|plts?|platelets?|Cr|creatinine|BUN|K|Na|glucose|Tbili|"
                    r"bilirubin|lactate|troponin|ferritin|fibrinogen|albumin|ALT|AST|ALP|CK)\s*[:=]?\s*(\d+(?:\.\d+)?)", re.I)


def clean(text):
    return PLACEHOLDER.sub(" ", str(text))


def extract_ground_truth(text):
    t = clean(text)
    low = t.lower()
    return {
        "numbers": sorted(set(NUM.findall(t)), key=float),
        "negations": sorted({p.replace(r"\b", "") for p in NEG_PATTERNS if re.search(p, low)}),
        "doses": [f"{m.group(1)} {m.group(2)} {m.group(3)}".lower() for m in DOSE_RE.finditer(t)],
        "labvalues": [f"{m.group(1)} {m.group(2)}".lower() for m in LAB_RE.finditer(t)],
        "flags": [w for w in FLAG_WORDS if re.search(r"\b" + w + r"\b", low)],
    }


def build_answer_key(df):
    """Return (dict snippet_id -> ground truth, DataFrame answer key)."""
    gt = {s: extract_ground_truth(t) for s, t in zip(df["SNIPPET_ID"], df["SNIPPET"])}
    key = pd.DataFrame([{"SNIPPET_ID": k, "GT_NUMBERS": "|".join(v["numbers"]),
                         "GT_NEGATIONS": "|".join(v["negations"]), "GT_DOSES": "|".join(v["doses"]),
                         "GT_LABVALUES": "|".join(v["labvalues"]), "GT_ABNORMAL_FLAGS": "|".join(v["flags"])}
                        for k, v in gt.items()])
    return gt, key
