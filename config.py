"""All settings in one place (same values as the report)."""

SEED = 42

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MT_MODEL = "facebook/nllb-200-distilled-600M"

LANG_CODES = {
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
}

TRANSLATE_ONLY = "translate_only_baseline"
SIMPLIFY_THEN_TRANSLATE = "simplify_then_translate"

SIMPLIFY_BATCH = 8
TRANSLATE_BATCH = 16
MAX_NEW_TOKENS = 300
MAX_LENGTH = 512

N_BOOT = 5000
CI_LEVEL = 0.95

SYSTEM_PROMPT = (
    "You rewrite clinical notes in simple, plain English for a patient. "
    "Do not add any new medical facts. Preserve every number, medication "
    "name, dose, and negation exactly."
)
