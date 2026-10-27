"""Score outputs you already have (no GPU needed): python score_existing.py --data final_evaluation_set.csv --bt pipeline_outputs_backtranslated.csv"""
import argparse

import pandas as pd

from medbridge.ground_truth import build_answer_key
from medbridge.scoring import score_all, summarize
from medbridge.stats import significance_table

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="final_evaluation_set.csv")
ap.add_argument("--bt", default="pipeline_outputs_backtranslated.csv", help="CSV with SNIPPET_ID, LANGUAGE, ORDER, OUTPUT, BACKTRANSLATION")
args = ap.parse_args()

df = pd.read_csv(args.data)[["SNIPPET_ID", "SNIPPET"]]
gt, _ = build_answer_key(df)
scored = score_all(pd.read_csv(args.bt), df, gt)
scored.to_csv("scored_outputs.csv", index=False)
print(summarize(scored).to_string(index=False))
print(significance_table(scored).to_string(index=False))
