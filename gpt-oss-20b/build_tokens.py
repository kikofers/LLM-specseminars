from transformers import AutoTokenizer
from wordfreq import top_n_list
import json
import re

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")

# Load 50k most common English words
words = top_n_list("en", n=50000)

# Filtering: keep alphabetic-only
allowed_pattern = re.compile(r"^[A-Za-z]+$")

clean_words = []

for w in words:
    if allowed_pattern.match(w):
        clean_words.append(w)

print("Words after cleaning:", len(clean_words))

token_dict = {}

# Tokenize every clean word
for word in clean_words:
    # Tokenize in lowercase (base form)
    tokens_lower = tokenizer.encode(word.lower(), add_special_tokens=False)
    if tokens_lower:
        token_dict[word.lower()] = tokens_lower

    # If original was uppercase acronym, store it too
    if word.isupper():
        tokens_upper = tokenizer.encode(word, add_special_tokens=False)
        if tokens_upper:
            token_dict[word] = tokens_upper

print("Total tokenized entries:", len(token_dict))

# Save as clean JSON dictionary
with open("english_token_dictionary_clean.json", "w", encoding="utf-8") as f:
    json.dump(token_dict, f, ensure_ascii=False, indent=2)

print("Saved dictionary to english_token_dictionary_clean.json")
