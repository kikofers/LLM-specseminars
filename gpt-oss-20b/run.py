import json
import random
import time
import sqlite3
import requests
from datetime import datetime
from tqdm import tqdm


# -----------------------------------------------------------
# Load configuration
# -----------------------------------------------------------

with open("config.json", "r") as f:
    CONFIG = json.load(f)

LMSTUDIO_URL = CONFIG["lmstudio_url"]
TEMPERATURE = CONFIG["temperature"]
MAX_TOKENS = CONFIG["max_output_tokens"]

DICT_PATH = CONFIG["dictionary_path"]
DB_PATH = CONFIG["database_path"]

SINGLE_ROUNDS = CONFIG["single_word_rounds"]
MULTI_ROUNDS = CONFIG["multi_word_rounds"]

TWO_WORD_P = CONFIG["multiword_mix"]["two_word_probability"]
THREE_WORD_P = CONFIG["multiword_mix"]["three_word_probability"]

RETRY_ATTEMPTS = CONFIG["retry_attempts"]
RETRY_DELAY = CONFIG["retry_delay_seconds"]


# -----------------------------------------------------------
# Load token dictionary
# -----------------------------------------------------------

print("Loading dictionary:", DICT_PATH)
with open(DICT_PATH, "r", encoding="utf-8") as f:
    WORD_DICT = json.load(f)

WORD_LIST = list(WORD_DICT.keys())


# -----------------------------------------------------------
# Database setup
# -----------------------------------------------------------

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    mode TEXT,
    target_words TEXT,
    target_tokens TEXT,
    scrambled_tokens TEXT,
    model_output TEXT,
    parsed_output TEXT,
    correct_words TEXT,
    missing_words TEXT,
    hallucinations TEXT,
    timestamp TEXT
);
""")

conn.commit()


# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

def pick_single_word_round():
    word = random.choice(WORD_LIST)
    tokens = WORD_DICT[word]

    scrambled = tokens[:]
    random.shuffle(scrambled)

    return {
        "mode": "single",
        "words": [word],
        "tokens": tokens,
        "scrambled": scrambled
    }


def pick_multi_word_round():
    # Decide 2-word or 3-word
    r = random.random()
    if r < TWO_WORD_P:
        count = 2
    else:
        count = 3

    words = random.sample(WORD_LIST, count)

    all_tokens = []
    for w in words:
        all_tokens.extend(WORD_DICT[w])

    scrambled = all_tokens[:]
    random.shuffle(scrambled)

    return {
        "mode": "multi",
        "words": words,
        "tokens": all_tokens,
        "scrambled": scrambled
    }


def call_lmstudio(scrambled_tokens):
    prompt = (
        "You are given a list of model token IDs:\n"
        f"{scrambled_tokens}\n\n"
        "Your task is to reconstruct all valid English words that can be formed "
        "from exactly these tokens. Return ONLY a JSON array of words, for example:\n"
        "[\"apple\", \"pear\"]\n\n"
        "Do not include any text outside the JSON list."
    )

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a strict evaluator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = requests.post(LMSTUDIO_URL, json=payload, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            return text
        except Exception as e:
            print(f"Request failed: {e}")
            if attempt + 1 == RETRY_ATTEMPTS:
                return "[]"
            time.sleep(RETRY_DELAY)


def parse_model_output(raw_text):
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed
        return []
    except:
        return []


def evaluate(words, parsed_output):
    correct = []
    missing = []
    halluc = []

    target_set = set(words)
    output_set = set(parsed_output)

    for w in target_set:
        if w in output_set:
            correct.append(w)
        else:
            missing.append(w)

    for w in output_set:
        if w not in target_set:
            halluc.append(w)

    return correct, missing, halluc


def save_result(round_data, model_output, parsed_output, correct, missing, halluc):
    cur.execute("""
        INSERT INTO results (
            mode, target_words, target_tokens, scrambled_tokens,
            model_output, parsed_output, correct_words, missing_words, hallucinations, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        round_data["mode"],
        json.dumps(round_data["words"]),
        json.dumps(round_data["tokens"]),
        json.dumps(round_data["scrambled"]),
        model_output,
        json.dumps(parsed_output),
        json.dumps(correct),
        json.dumps(missing),
        json.dumps(halluc),
        datetime.now().isoformat()
    ))
    conn.commit()


# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------

print("\n=== Phase 1: Single-word rounds ===")
for _ in tqdm(range(SINGLE_ROUNDS)):
    rd = pick_single_word_round()
    raw = call_lmstudio(rd["scrambled"])
    parsed = parse_model_output(raw)
    correct, missing, halluc = evaluate(rd["words"], parsed)
    save_result(rd, raw, parsed, correct, missing, halluc)


print("\n=== Phase 2: Multi-word rounds ===")
for _ in tqdm(range(MULTI_ROUNDS)):
    rd = pick_multi_word_round()
    raw = call_lmstudio(rd["scrambled"])
    parsed = parse_model_output(raw)
    correct, missing, halluc = evaluate(rd["words"], parsed)
    save_result(rd, raw, parsed, correct, missing, halluc)


print("\nExperiment completed.\nResults stored in:", DB_PATH)
conn.close()
