"""The two pipeline orderings and the back-translation step."""
import pandas as pd

from . import config as C


def run_pipelines(df, simplifier, translator, out_dir=None, log=print):
    """Return a table with one row per snippet x language x ordering (250 x 6 x 2 = 3000)."""
    ids, snips = df["SNIPPET_ID"].tolist(), df["SNIPPET"].tolist()
    simplified = simplifier(snips)
    if out_dir:
        pd.DataFrame({"SNIPPET_ID": ids, "SIMPLIFIED_TEXT": simplified}).to_csv(f"{out_dir}/simplified_texts.csv", index=False)
    rows = []
    for lang, code in C.LANG_CODES.items():
        direct = translator(snips, "eng_Latn", code)          # translate-only
        chained = translator(simplified, "eng_Latn", code)    # simplify, then translate
        rows += [{"SNIPPET_ID": s, "LANGUAGE": lang, "ORDER": C.TRANSLATE_ONLY, "OUTPUT": o} for s, o in zip(ids, direct)]
        rows += [{"SNIPPET_ID": s, "LANGUAGE": lang, "ORDER": C.SIMPLIFY_THEN_TRANSLATE, "OUTPUT": o} for s, o in zip(ids, chained)]
        if out_dir:
            pd.DataFrame(rows).to_csv(f"{out_dir}/pipeline_outputs.csv", index=False)
        log(f"{lang} done")
    return pd.DataFrame(rows)


def back_translate(outputs, translator, repetition_penalty=1.0, languages=None, log=print):
    """Translate every output back to English (used to check negations and abnormal flags)."""
    res = outputs.copy()
    res["BACKTRANSLATION"] = ""
    for lang, code in C.LANG_CODES.items():
        if languages and lang not in languages:
            continue
        m = res["LANGUAGE"] == lang
        res.loc[m, "BACKTRANSLATION"] = translator(res.loc[m, "OUTPUT"].tolist(), code, "eng_Latn",
                                                   repetition_penalty=repetition_penalty)
        log(f"back-translated {lang}")
    return res
