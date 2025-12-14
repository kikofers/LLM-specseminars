import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from collections import Counter

from openai import OpenAI


BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "openai/gpt-oss-20b"

# Benchmark controls
N_ROUNDS = 20
K_MAX = 20

# Response budget controls (tune these for speed)
TEMPERATURE = 0.0
MAX_TOKENS = 800  # keep low for speed; raise if recall is too low

ROUNDS_JSONL = r".\rounds\rounds_5000_sp_with_solutions.jsonl"

SYSTEM_PROMPT = f"""Output ONLY a valid JSON list of up to {K_MAX} constructions from the tiles. No reasoning, explanations, or extra text.

Rules:
1. used_tokens: subset of the provided tiles (use exact strings, respect counts, no extras or modifications).
2. concat: exactly ''.join(used_tokens). Example: used_tokens=["Ġhello", "world"] -> concat="Ġhelloworld". For tiles=["a","b"], used_tokens=["a","b"] -> concat="ab".
3. word: concat with 'Ġ'->' ', 'Ċ'->'\\n', then strip and lowercase. Example: "Ġhelloworld" -> "helloworld". "ab" -> "ab".
4. Objects: {{"word": str, "used_tokens": [str, ...], "concat": str}}
5. If none, output [].
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "constructions_idx",
        "strict": True,
        "schema": {
            "type": "array",
            "maxItems": K_MAX,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idx"],
                "properties": {
                    "idx": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1
                    }
                }
            }
        }
    }
}

def normalize_concat_to_word(concat: str) -> str:
    return concat.replace("Ġ", " ").replace("Ċ", "\n").strip().lower()


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def call_model(client: OpenAI, tiles: List[str]) -> Tuple[List[Dict[str, Any]], float, str]:
    user_payload = {"tiles": tiles}
    user_text = json.dumps(user_payload, ensure_ascii=False)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format=RESPONSE_FORMAT,
    )
    dt = time.perf_counter() - t0

    raw = resp.choices[0].message.content or ""
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed)}. Raw:\n{raw}")
    return parsed, dt, raw


def evaluate_outputs(
    outputs: List[Dict[str, Any]],
    tiles: List[str],
    all_solutions: Set[str],
    targets: Set[str],
) -> Dict[str, Any]:
    fmt_err = 0
    pred_words = set()

    index_oob = 0
    index_reuse = 0

    for it in outputs[:K_MAX]:
        if not isinstance(it, dict) or "idx" not in it or not isinstance(it["idx"], list):
            fmt_err += 1
            continue
        if any(not isinstance(x, int) for x in it["idx"]):
            fmt_err += 1
            continue

        idx_list = it["idx"]

        # bounds check
        if any(x < 0 or x >= len(tiles) for x in idx_list):
            index_oob += 1
            continue

        # multiplicity check (each position usable once)
        if len(set(idx_list)) != len(idx_list):
            index_reuse += 1
            continue

        used_tokens = [tiles[i] for i in idx_list]
        concat = "".join(used_tokens)
        w = normalize_concat_to_word(concat)
        pred_words.add(w)

    valid_hits = pred_words.intersection(all_solutions)
    target_hits = pred_words.intersection(targets)

    n_pred = len(pred_words)
    n_valid = len(valid_hits)
    n_solutions = max(1, len(all_solutions))
    n_targets = max(1, len(targets))

    precision = (n_valid / n_pred) if n_pred else 0.0
    full_recall = (n_valid / n_solutions) if n_solutions else 0.0
    target_recall = (len(target_hits) / n_targets) if n_targets else 0.0

    return {
        "n_output_items_raw": len(outputs),
        "n_pred_unique": n_pred,
        "n_valid_unique": n_valid,
        "n_solutions": len(all_solutions),
        "n_targets": len(targets),
        "precision": precision,
        "full_recall": full_recall,
        "target_recall": target_recall,
        "format_errors": fmt_err,
        "index_oob": index_oob,
        "index_reuse": index_reuse,
    }



def main():
    client = OpenAI(base_url=BASE_URL, api_key="lm-studio")

    rows = []
    rounds_iter = iter_jsonl(ROUNDS_JSONL)

    for i in range(N_ROUNDS):
        r = next(rounds_iter)
        round_id = r.get("round_id", i)
        tiles = r["tiles"]
        all_solutions = set(r.get("all_solutions", []))
        targets = set(r.get("targets", []))

        try:
            outputs, dt, raw = call_model(client, tiles)
            print("RAW MODEL JSON:", raw)
            metrics = evaluate_outputs(outputs, tiles, all_solutions, targets)
            metrics.update({
                "round_id": round_id,
                "latency_ms": int(dt * 1000),
            })
        except Exception as e:
            metrics = {
                "round_id": round_id,
                "latency_ms": None,
                "error": str(e),
            }

        rows.append(metrics)
        print(f"Round {i+1}/{N_ROUNDS} (id={round_id}) -> {metrics.get('latency_ms')} ms | "
              f"prec={metrics.get('precision')} recall={metrics.get('full_recall')} target={metrics.get('target_recall')} "
              f"err={metrics.get('error', '')}")

    # Aggregate summary (ignore errored rows)
    ok = [r for r in rows if "error" not in r]
    if ok:
        avg_latency = sum(r["latency_ms"] for r in ok if r["latency_ms"] is not None) / len(ok)
        avg_prec = sum(r["precision"] for r in ok) / len(ok)
        avg_full = sum(r["full_recall"] for r in ok) / len(ok)
        avg_target = sum(r["target_recall"] for r in ok) / len(ok)
        print("\n=== Summary (successful rounds) ===")
        print(f"Rounds: {len(ok)}/{len(rows)}")
        print(f"Avg latency: {avg_latency:.1f} ms")
        print(f"Avg precision: {avg_prec:.3f}")
        print(f"Avg full recall: {avg_full:.3f}")
        print(f"Avg target recall: {avg_target:.3f}")


if __name__ == "__main__":
    main()
