# MedBridge

Faithfulness and safety of LLM-based clinical report simplification and translation into Indian languages.

The study checks whether numbers, negations and abnormal flags in a clinical snippet survive two pipelines:
1. **Translate-only:** English snippet translated directly with NLLB-200 (distilled 600M).
2. **Simplify-then-translate:** the snippet is first simplified in English with Qwen2.5-3B-Instruct, then translated.

Languages: Telugu, Tamil, Kannada, Hindi, Bengali, Marathi. Data: 250 de-identified MIMIC-III snippets (access needs PhysioNet credentialing, so the raw text is not in this repo).

## Structure
| File | Purpose |
|---|---|
| `medbridge/config.py` | models, languages, batch sizes, seed |
| `medbridge/models.py` | Qwen simplifier and NLLB translator (greedy decoding) |
| `medbridge/pipeline.py` | the two orderings and back-translation |
| `medbridge/ground_truth.py` | automatic answer key |
| `medbridge/scoring.py` | numeric fidelity, hallucinated numbers, negation and flag survival |
| `medbridge/stats.py` | paired sign test and bootstrap confidence interval |
| `run_experiment.py` | full run (needs a GPU) |
| `score_existing.py` | score existing outputs (no GPU) |

## Run (Google Colab, T4 GPU)
```
!git clone https://github.com/<team-username>/medbridge.git
%cd medbridge
!pip install -q -r requirements.txt
# upload final_evaluation_set.csv into this folder, then:
!python run_experiment.py --data final_evaluation_set.csv --ablation
```
Results are written to `medbridge_outputs/`. A full run takes under about 50 minutes on a T4.

## Scoring without a GPU
```
python score_existing.py --data final_evaluation_set.csv --bt pipeline_outputs_backtranslated.csv
```

## Notes
- Decoding is greedy (no sampling, no beam search) with seed 42.
- Negation and flag survival use round-trip back-translation, so the numbers are a conservative lower bound.
