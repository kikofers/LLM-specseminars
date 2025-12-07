from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import json

# Download only tokenizer-related files
snapshot_download(
    repo_id="openai/gpt-oss-20b",
    local_dir="gpt-oss-20b",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",          # included just in case
        "*.txt",
    ]
)

print("Tokenizer files downloaded.")

# Load from local directory
tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")

# Extract vocab (token → ID)
vocab = tokenizer.get_vocab()

print("Vocabulary size:", len(vocab))

# Save vocabulary to JSON file
with open("gpt_oss_20b_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("Saved vocabulary to gpt_oss_20b_vocab.json")
