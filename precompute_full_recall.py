"""
To run the code, paste the command below into your terminal:

python precompute_full_recall.py ^
  --dict .\jsons\english_token_dictionary_bow_sp.json ^
  --rounds .\rounds\rounds_5000_sp.jsonl ^
  --out .\rounds\rounds_5000_sp_with_solutions.jsonl ^
  --variant sp ^
  --verify_round_variant

"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Iterable, Set


def load_dictionary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def is_entry_ok(entry: Any, variant: str) -> bool:
    if not isinstance(entry, dict):
        return False
    v = entry.get(variant)
    if not isinstance(v, dict):
        return False
    toks = v.get("tokens")
    return isinstance(toks, list) and len(toks) > 0 and all(isinstance(t, str) for t in toks)


def build_indices(dictionary: Dict[str, Any], variant: str):
    """
    Returns:
      - word_list: list of words
      - word_token_counters: list of Counter(token->count) aligned with word_list
      - token_to_word_ids: dict token -> list[int] of word indices containing that token
    """
    word_list: List[str] = []
    word_token_counters: List[Counter] = []
    token_to_word_ids: Dict[str, List[int]] = defaultdict(list)

    for w, e in dictionary.items():
        if not is_entry_ok(e, variant):
            continue
        toks = e[variant]["tokens"]
        c = Counter(toks)

        idx = len(word_list)
        word_list.append(w)
        word_token_counters.append(c)

        # Reverse index: token -> candidate words containing that token
        for t in c.keys():
            token_to_word_ids[t].append(idx)

    return word_list, word_token_counters, token_to_word_ids


def multiset_subset(needed: Counter, available: Counter) -> bool:
    # True iff every token count needed[t] <= available[t]
    for t, n in needed.items():
        if available.get(t, 0) < n:
            return False
    return True


def compute_solutions_for_round(
    tiles: List[str],
    word_list: List[str],
    word_token_counters: List[Counter],
    token_to_word_ids: Dict[str, List[int]],
) -> List[str]:
    tiles_counter = Counter(tiles)

    # Candidate set: union of words that share at least one token with tiles.
    # This avoids scanning the entire 50k list for every round.
    candidate_ids: Set[int] = set()
    for t in tiles_counter.keys():
        ids = token_to_word_ids.get(t)
        if ids:
            candidate_ids.update(ids)

    solutions: List[str] = []
    for idx in candidate_ids:
        if multiset_subset(word_token_counters[idx], tiles_counter):
            solutions.append(word_list[idx])

    solutions.sort()
    return solutions


def main():
    ap = argparse.ArgumentParser(description="Add full-recall solution sets to precomputed rounds JSONL.")
    ap.add_argument("--dict", required=True, help="Path to english_token_dictionary_bow_sp.json")
    ap.add_argument("--rounds", required=True, help="Input rounds JSONL (precomputed)")
    ap.add_argument("--out", required=True, help="Output JSONL with full recall fields added")
    ap.add_argument("--variant", choices=["sp", "bow"], default="sp",
                    help="Which tokenisation variant to use for solutions (should match round.variant)")
    ap.add_argument("--verify_round_variant", action="store_true",
                    help="If set, checks each round['variant'] equals --variant and raises if not.")
    args = ap.parse_args()

    dictionary = load_dictionary(args.dict)

    word_list, word_token_counters, token_to_word_ids = build_indices(dictionary, args.variant)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for r in iter_jsonl(args.rounds):
            if args.verify_round_variant:
                rv = r.get("variant")
                if rv != args.variant:
                    raise ValueError(f"Round variant mismatch: round has {rv}, expected {args.variant}")

            tiles = r.get("tiles")
            if not isinstance(tiles, list) or any(not isinstance(t, str) for t in tiles):
                raise ValueError(f"Invalid tiles in round_id={r.get('round_id')}")

            solutions = compute_solutions_for_round(
                tiles=tiles,
                word_list=word_list,
                word_token_counters=word_token_counters,
                token_to_word_ids=token_to_word_ids,
            )

            # Add fields
            r["all_solutions"] = solutions
            r["n_solutions"] = len(solutions)

            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rounds with full recall to: {out_path}")


if __name__ == "__main__":
    main()
