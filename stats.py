"""Paired sign test and bootstrap confidence interval, in plain Python (no stats packages)."""
import math
import random

import pandas as pd

from . import config as C


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def sign_test(diffs):
    """Two-sided sign test with normal approximation. Returns the p-value."""
    pos = sum(1 for d in diffs if d > 0)
    n = pos + sum(1 for d in diffs if d < 0)
    if n == 0:
        return None
    z = (pos - n / 2) / math.sqrt(n / 4)
    return 2 * (1 - normal_cdf(abs(z)))


def bootstrap_ci(diffs, n_resamples=C.N_BOOT, ci=C.CI_LEVEL, seed=C.SEED):
    """Bootstrap interval for the mean paired difference."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_resamples))
    return means[int((1 - ci) / 2 * n_resamples)], means[int((1 + ci) / 2 * n_resamples) - 1]


def paired_diffs(scored, col, lang):
    """Translate-only minus simplify-then-translate, per snippet (positive = translate-only better)."""
    sub = scored[scored.LANGUAGE == lang]
    a = sub[sub.ORDER == C.TRANSLATE_ONLY].set_index("SNIPPET_ID")[col]
    b = sub[sub.ORDER == C.SIMPLIFY_THEN_TRANSLATE].set_index("SNIPPET_ID")[col]
    j = pd.concat([a, b], axis=1, keys=["t", "s"]).dropna().astype(float)
    return (j["t"] - j["s"]).tolist()


def significance_table(scored, n_resamples=C.N_BOOT):
    metrics = [("NUMERIC_FIDELITY_PCT", "numeric_fidelity", 1),
               ("NEGATION_PRESERVED", "negation_survival", 100),
               ("FLAG_PRESERVED", "flag_survival", 100)]
    rows = []
    for lang in C.LANG_CODES:
        for col, name, scale in metrics:
            d = paired_diffs(scored, col, lang)
            if not d:
                continue
            lo, hi = bootstrap_ci(d, n_resamples)
            rows.append({"LANGUAGE": lang, "METRIC": name, "N": len(d),
                         "MEAN_GAP_POINTS": round(sum(d) / len(d) * scale, 1),
                         "CI95_LOW": round(lo * scale, 1), "CI95_HIGH": round(hi * scale, 1),
                         "SIGN_TEST_P": sign_test(d)})
    return pd.DataFrame(rows)
