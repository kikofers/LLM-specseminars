from collections import Counter
from typing import Any, Dict, List


def normalize_concat_to_word(concat: str) -> str:
    # OpenAI/GPT-style BPE markers commonly used in token strings
    # Ġ = space, Ċ = newline
    s = concat.replace("Ġ", " ").replace("Ċ", "\n")
    return s.strip().lower()


def evaluate_round(
    model_outputs: List[Dict[str, Any]],
    tile_tokens: List[str],
    dictionary: Dict[str, Any],
    variant: str = "sp",
) -> Dict[str, Any]:
    tiles_counter = Counter(tile_tokens)

    details = []
    stats = Counter()

    for item in model_outputs:
        word = item.get("word")
        used = item.get("used_tokens")
        concat_field = item.get("concat")

        # Format checks
        if (
            not isinstance(word, str)
            or not isinstance(concat_field, str)
            or not isinstance(used, list)
            or any(not isinstance(t, str) for t in used)
        ):
            stats["format_error"] += 1
            details.append({"item": item, "status": "format_error"})
            continue

        word_norm = word.strip().lower()

        # Tile availability (multiset)
        used_counter = Counter(used)
        if any(used_counter[t] > tiles_counter.get(t, 0) for t in used_counter):
            stats["hallucinated_token_or_overuse"] += 1
            details.append({"word": word, "used_tokens": used, "concat": concat_field,
                            "status": "hallucinated_token_or_overuse"})
            continue

        # Exact concatenation must match
        concat_exact = "".join(used)
        if concat_field != concat_exact:
            stats["concat_field_mismatch"] += 1
            details.append({"word": word, "used_tokens": used, "concat": concat_field,
                            "status": "concat_field_mismatch", "expected_concat": concat_exact})
            continue

        # Normalized surface word must match concat
        expected_word = normalize_concat_to_word(concat_exact)
        if word_norm != expected_word:
            stats["word_normalization_mismatch"] += 1
            details.append({"word": word, "used_tokens": used, "concat": concat_field,
                            "status": "word_normalization_mismatch", "expected_word": expected_word})
            continue

        # Dictionary membership (use normalized word)
        entry = dictionary.get(word_norm)
        if entry is None:
            stats["not_in_dictionary"] += 1
            details.append({"word": word_norm, "used_tokens": used, "concat": concat_field,
                            "status": "not_in_dictionary"})
            continue

        # Canonical tokenisation match (variant-aware)
        canonical_tokens = ((entry.get(variant) or {}).get("tokens")) if isinstance(entry, dict) else None
        if canonical_tokens is None:
            stats["no_canonical_variant"] += 1
            details.append({"word": word_norm, "used_tokens": used, "concat": concat_field,
                            "status": "no_canonical_variant"})
            continue

        if used == canonical_tokens:
            stats["correct_canonical"] += 1
            details.append({"word": word_norm, "used_tokens": used, "concat": concat_field,
                            "status": "correct_canonical"})
        else:
            stats["alt_tokenisation"] += 1
            details.append({"word": word_norm, "used_tokens": used, "concat": concat_field,
                            "status": "alt_tokenisation", "canonical": canonical_tokens})

    return {"summary": dict(stats), "details": details}
