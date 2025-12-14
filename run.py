import json
import random

from lmstudio_client import LMStudioClient
from evaluator import evaluate_round


SYSTEM_PROMPT = """You are playing a token-tile anagram game.

Rules:
- Each tile is an atomic token string. Do not alter, split, merge, or edit tiles.
- You may only use the provided tiles, each at most as many times as it appears.
- A candidate is valid only if concat == exact concatenation of used_tokens (in order).
- The field 'word' must be the normalized surface form of concat:
  - replace 'Ġ' with a space
  - replace 'Ċ' with a newline
  - then strip leading/trailing whitespace
  - then lowercase
- Output MUST be valid JSON (a list of objects).
- Each object MUST have exactly these keys: {"word": string, "used_tokens": [string, ...], "concat": string}
- Do not include explanations, markdown, or extra text.
"""


def load_dictionary(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sample_tiles_from_dictionary(dictionary, k_words=3, distractors=6, variant="sp"):
    """
    Very simple sampler:
    - chooses k_words random words from dictionary with available variant tokens
    - tiles are union of their tokens + distractor tokens from other random words
    """
    candidates = [w for w, e in dictionary.items() if isinstance(e, dict) and e.get(variant) and e[variant].get("tokens")]
    chosen_words = random.sample(candidates, k_words)

    tiles = []
    for w in chosen_words:
        tiles.extend(dictionary[w][variant]["tokens"])

    # Add distractor tiles sampled from random other words
    for _ in range(distractors):
        w = random.choice(candidates)
        toks = dictionary[w][variant]["tokens"]
        tiles.append(random.choice(toks))

    random.shuffle(tiles)
    return chosen_words, tiles


def main():
    DICT_PATH = "./jsons/english_token_dictionary_bow_sp.json"
    dictionary = load_dictionary(DICT_PATH)

    chosen_words, tiles = sample_tiles_from_dictionary(dictionary, k_words=3, distractors=8, variant="sp")

    user_payload = {
        "tiles": tiles,
        "task": "Find as many valid dictionary words as you can form from the tiles.",
        "output_schema": [{"word": "string", "used_tokens": ["string"], "concat": "string"}],
        "constraints": {"lowercase_only": True, "max_candidates": 50}
    }

    client = LMStudioClient(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # You may need to set this to the exact model name LM Studio exposes
    MODEL_NAME = "openai/gpt-oss-20b"

    outputs = client.chat_json(
        model=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        user_payload=user_payload,
        temperature=0.0,
        max_tokens=1200,
    )

    report = evaluate_round(outputs, tiles, dictionary, variant="sp")

    print("Ground-truth included words (not revealed to model):", chosen_words)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))

    # Optional: print first few details
    for d in report["details"][:10]:
        print(d)


if __name__ == "__main__":
    main()
