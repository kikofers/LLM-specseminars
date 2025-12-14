from transformers import AutoTokenizer
from wordfreq import top_n_list
import json
import re
from collections import Counter
import os

# ----------------------------
# Configuration
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "jsons")  # your local tokenizer folder
OUT_PATH = os.path.join(SCRIPT_DIR, "jsons", "english_token_dictionary_bow_sp.json")
N_WORDS = 50000

# Keep alphabetic-only English words (no hyphens, apostrophes, etc.)
ALLOWED_PATTERN = re.compile(r"^[A-Za-z]+$")

# If True, store only lowercase keys for normal words.
# Acronyms (ALLCAPS) will also be stored under original casing.
FORCE_LOWERCASE = True


def encode_variant(tokenizer, text: str):
    """
    Returns a dict with ids and tokens, or None if empty.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return None
    toks = tokenizer.convert_ids_to_tokens(ids)
    return {"ids": ids, "tokens": toks}


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Load top English words
    words = top_n_list("en", n=N_WORDS)

    # Filter alphabetic-only
    clean_words = [w for w in words if ALLOWED_PATTERN.match(w)]
    print("Words after cleaning:", len(clean_words))

    out = {}
    skipped = 0

    for w in clean_words:
        # Normalize key
        key = w
        if FORCE_LOWERCASE and not w.isupper():
            key = w.lower()

        # Build variants
        bow = encode_variant(tokenizer, key)
        sp = encode_variant(tokenizer, " " + key)

        if bow is None and sp is None:
            skipped += 1
            continue

        out[key] = {
            "bow": bow,
            "sp": sp,
        }

        # If original was uppercase acronym and you want it too
        if w.isupper():
            bow_u = encode_variant(tokenizer, w)
            sp_u = encode_variant(tokenizer, " " + w)
            if bow_u or sp_u:
                out[w] = {"bow": bow_u, "sp": sp_u}

    print("Total entries:", len(out))
    print("Skipped:", skipped)

    # Save JSON
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved to:", OUT_PATH)

    # Show a sample
    sample_key = next(iter(out.keys()))
    print("Sample key:", sample_key)
    print("Sample value:", json.dumps(out[sample_key], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
