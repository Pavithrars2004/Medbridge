"""Run the full MedBridge experiment: python run_experiment.py --data final_evaluation_set.csv"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd

from medbridge import config as C
from medbridge.ground_truth import build_answer_key
from medbridge.scoring import score_all, summarize
from medbridge.stats import significance_table


def main():
    ap = argparse.ArgumentParser(description="MedBridge faithfulness study")
    ap.add_argument("--data", default="final_evaluation_set.csv", help="CSV with SNIPPET_ID and SNIPPET columns")
    ap.add_argument("--out", default="medbridge_outputs", help="output folder")
    ap.add_argument("--ablation", action="store_true", help="also run the Telugu repetition-penalty 1.3 ablation")
    args = ap.parse_args()

    import torch
    from medbridge.models import Simplifier, Translator
    from medbridge.pipeline import back_translate, run_pipelines

    assert torch.cuda.is_available(), "A GPU is needed (Colab: Runtime > Change runtime type > T4 GPU)"
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(args.data)[["SNIPPET_ID", "SNIPPET"]].dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} snippets")
    gt, key = build_answer_key(df)
    key.to_csv(f"{args.out}/ground_truth_answer_key.csv", index=False)

    simplifier, translator = Simplifier(), Translator()
    outputs = run_pipelines(df, simplifier, translator, args.out)
    bt = back_translate(outputs, translator)
    bt.to_csv(f"{args.out}/pipeline_outputs_backtranslated.csv", index=False)

    scored = score_all(bt, df, gt)
    scored.to_csv(f"{args.out}/scored_outputs.csv", index=False)
    summarize(scored).to_csv(f"{args.out}/results_summary.csv", index=False)
    significance_table(scored).to_csv(f"{args.out}/statistical_significance.csv", index=False)

    if args.ablation:
        tel = outputs[outputs.LANGUAGE == "Telugu"]
        s13 = score_all(back_translate(tel, translator, 1.3, ["Telugu"]), df, gt)
        s10 = scored[scored.LANGUAGE == "Telugu"]
        pd.DataFrame({"setting": ["repetition_penalty 1.0", "repetition_penalty 1.3"],
                      "negation_survival": [x.NEGATION_PRESERVED.dropna().astype(float).mean() * 100 for x in (s10, s13)],
                      "flag_survival": [x.FLAG_PRESERVED.dropna().astype(float).mean() * 100 for x in (s10, s13)]}
                     ).round(1).to_csv(f"{args.out}/ablation_repetition_penalty_telugu.csv", index=False)

    print(f"Done in {(time.time() - t0) / 60:.1f} minutes. Results in {args.out}/")


if __name__ == "__main__":
    main()
