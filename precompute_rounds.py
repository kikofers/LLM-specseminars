"""
To run the code, paste the command below into your terminal:

python precompute_rounds.py ^
  --dict .\jsons\english_token_dictionary_bow_sp.json ^
  --out  .\rounds\rounds_5000_sp.jsonl ^
  --n_rounds 5000 ^
  --variant sp ^
  --k_targets 3 ^
  --distractors 8 ^
  --min_tokens 1 ^
  --max_tokens 4 ^
  --seed 12345 ^
  --ensure_unique_rounds

"""


import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter


def load_dictionary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_entry_ok(entry: Any, variant: str) -> bool:
    if not isinstance(entry, dict):
        return False
    v = entry.get(variant)
    if not isinstance(v, dict):
        return False
    toks = v.get("tokens")
    return isinstance(toks, list) and len(toks) > 0 and all(isinstance(t, str) for t in toks)


def build_word_pool(dictionary: Dict[str, Any], variant: str, min_tokens: int, max_tokens: int) -> List[str]:
    pool = []
    for w, e in dictionary.items():
        if not is_entry_ok(e, variant):
            continue
        n = len(e[variant]["tokens"])
        if min_tokens <= n <= max_tokens:
            pool.append(w)
    return pool


def pick_targets(
    pool: List[str],
    dictionary: Dict[str, Any],
    variant: str,
    k_targets: int,
    max_attempts: int = 2000,
) -> List[str]:
    """
    Pick k_targets words. Attempts to avoid duplicates and to prefer variety in token sequences.
    """
    if len(pool) < k_targets:
        raise ValueError("Pool too small for requested number of targets.")

    for _ in range(max_attempts):
        targets = random.sample(pool, k_targets)
        # Optional: you can add additional constraints here (e.g., disallow identical first token).
        return targets

    raise RuntimeError("Failed to pick targets under constraints.")


def sample_distractor_tokens(
    pool: List[str],
    dictionary: Dict[str, Any],
    variant: str,
    n_distractors: int,
) -> List[str]:
    distractors = []
    for _ in range(n_distractors):
        w = random.choice(pool)
        toks = dictionary[w][variant]["tokens"]
        distractors.append(random.choice(toks))
    return distractors


def round_tiles_from_targets(
    targets: List[str],
    dictionary: Dict[str, Any],
    variant: str,
    n_distractors: int,
    pool_for_distractors: List[str],
    shuffle_tiles: bool = True,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns:
      tiles: list of token strings (multiset)
      target_map: {word: [token strings used for that word in chosen variant]}
    """
    tiles: List[str] = []
    target_map: Dict[str, List[str]] = {}

    for w in targets:
        toks = dictionary[w][variant]["tokens"]
        target_map[w] = toks
        tiles.extend(toks)

    tiles.extend(sample_distractor_tokens(pool_for_distractors, dictionary, variant, n_distractors))

    if shuffle_tiles:
        random.shuffle(tiles)

    return tiles, target_map


def main():
    ap = argparse.ArgumentParser(description="Precompute token-tile anagram rounds (JSONL).")
    ap.add_argument("--dict", required=True, help="Path to english_token_dictionary_bow_sp.json")
    ap.add_argument("--out", required=True, help="Output JSONL path, e.g. rounds_5000.jsonl")
    ap.add_argument("--n_rounds", type=int, default=5000)
    ap.add_argument("--variant", choices=["sp", "bow"], default="sp")
    ap.add_argument("--k_targets", type=int, default=3, help="How many guaranteed solvable target words per round")
    ap.add_argument("--distractors", type=int, default=8, help="How many distractor tokens per round")
    ap.add_argument("--seed", type=int, default=12345)

    # Control difficulty by token length of target words
    ap.add_argument("--min_tokens", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=4)

    # Safety valves
    ap.add_argument("--max_tiles", type=int, default=80, help="Hard cap to avoid huge tile sets")
    ap.add_argument("--ensure_unique_rounds", action="store_true", help="Avoid exact duplicate tile-multisets")

    args = ap.parse_args()

    random.seed(args.seed)

    dictionary = load_dictionary(args.dict)

    pool = build_word_pool(dictionary, args.variant, args.min_tokens, args.max_tokens)
    if not pool:
        raise RuntimeError("No eligible words found. Adjust --variant/--min_tokens/--max_tokens.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_signatures = set()

    written = 0
    attempts = 0
    max_attempts = args.n_rounds * 20  # avoid infinite loops with strict constraints

    with open(out_path, "w", encoding="utf-8") as f:
        while written < args.n_rounds and attempts < max_attempts:
            attempts += 1

            targets = pick_targets(pool, dictionary, args.variant, args.k_targets)
            tiles, target_map = round_tiles_from_targets(
                targets=targets,
                dictionary=dictionary,
                variant=args.variant,
                n_distractors=args.distractors,
                pool_for_distractors=pool,
                shuffle_tiles=True,
            )

            if len(tiles) > args.max_tiles:
                continue

            # Optional: ensure rounds are unique by multiset signature
            if args.ensure_unique_rounds:
                sig = tuple(sorted(Counter(tiles).items()))
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

            round_obj = {
                "round_id": written,
                "variant": args.variant,
                "tiles": tiles,
                "targets": list(target_map.keys()),
                "target_tokens": target_map,  # explicit ground truth token sequences
                "meta": {
                    "k_targets": args.k_targets,
                    "distractors": args.distractors,
                    "min_tokens": args.min_tokens,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                },
            }

            f.write(json.dumps(round_obj, ensure_ascii=False) + "\n")
            written += 1

    if written < args.n_rounds:
        raise RuntimeError(f"Only wrote {written}/{args.n_rounds} rounds; relax constraints or increase max_attempts.")

    print(f"Wrote {written} rounds to: {out_path}")


if __name__ == "__main__":
    main()
