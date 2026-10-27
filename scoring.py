"""Faithfulness scoring: numeric fidelity, hallucinated numbers, negation and flag survival."""
import pandas as pd

from .ground_truth import NUM, clean

NEG_CUES_EN = ["no evidence of", "denies", "denied", "without", "negative for", "absence of",
               "no ", "not ", "never", "none"]
FLAG_CUES_EN = ["elevated", "decreased", "abnormal", "critical", "high", "low",
                "increased", "reduced", "raised", "lowered"]

# Native digits (Devanagari, Bengali, Tamil, Telugu, Kannada) -> Western digits
DIGIT_MAP = {base + i: ord("0") + i for base in (0x0966, 0x09E6, 0x0BE6, 0x0C66, 0x0CE6) for i in range(10)}


def normalize_digits(text):
    return str(text).translate(DIGIT_MAP)


def numbers_in(text):
    return set(NUM.findall(normalize_digits(clean(text))))


def numeric_scores(source_text, output_text):
    """Return (fidelity %, hallucinated numbers, has hallucination) for one output."""
    src, out = numbers_in(source_text), numbers_in(output_text)
    if not src:
        return None, [], False
    halluc = sorted(out - src)
    return len(src & out) / len(src) * 100, halluc, len(halluc) > 0


def cue_survives(back_text, cues):
    low = str(back_text).lower()
    return any(c in low for c in cues)


def score_all(bt_df, df, gt):
    """Score every back-translated output against its source snippet and ground truth."""
    src = dict(zip(df["SNIPPET_ID"], df["SNIPPET"]))
    rec = []
    for r in bt_df.itertuples():
        g = gt[r.SNIPPET_ID]
        fid, halluc, has_h = numeric_scores(src[r.SNIPPET_ID], r.OUTPUT)
        rec.append({
            "SNIPPET_ID": r.SNIPPET_ID, "LANGUAGE": r.LANGUAGE, "ORDER": r.ORDER,
            "NUMERIC_FIDELITY_PCT": fid, "HALLUCINATED_NUMBERS": "|".join(halluc),
            "HAS_HALLUCINATED_NUMBER": has_h,
            "NEGATION_PRESERVED": cue_survives(r.BACKTRANSLATION, NEG_CUES_EN) if g["negations"] else None,
            "FLAG_PRESERVED": cue_survives(r.BACKTRANSLATION, FLAG_CUES_EN) if g["flags"] else None,
        })
    return pd.DataFrame(rec)


def summarize(scored):
    """Mean scores per language and ordering."""
    return (scored.groupby(["LANGUAGE", "ORDER"])
            .agg(n=("SNIPPET_ID", "count"),
                 numeric_fidelity=("NUMERIC_FIDELITY_PCT", "mean"),
                 hallucinated_number_rate=("HAS_HALLUCINATED_NUMBER", lambda s: s.mean() * 100),
                 negation_survival=("NEGATION_PRESERVED", lambda s: s.dropna().astype(float).mean() * 100),
                 flag_survival=("FLAG_PRESERVED", lambda s: s.dropna().astype(float).mean() * 100))
            .round(1).reset_index())
