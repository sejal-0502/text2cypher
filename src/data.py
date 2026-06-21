# data.py
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_PROMPT = "You are a Cypher query generator. Given a graph schema and a natural language question, generate the correct Cypher query. Output only the Cypher query, nothing else."

def load_splits():
    """Load the dataset from HF and return the splits."""

    ds = load_dataset("RomanTeucher/text2cypher-curated")
    return ds['train'], ds['val'], ds['test']


def format_prompt(example, include_answer=True):
    """Format a single example into the chat template."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema: {example['schema']}\nQuestion: {example['question']}"},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": example['cypher']})
    return messages

def tokenize_example(example, tokenizer, max_length=512):
    """Tokenize and create labels with prompt tokens masked."""
    
    # Full sequence (prompt + cypher)
    messages = format_prompt(example, include_answer=True)
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Prompt only (to know where cypher starts)
    prompt_messages = format_prompt(example, include_answer=False)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize both
    full_tokens = tokenizer(full_text, max_length=max_length, truncation=True)
    prompt_tokens = tokenizer(prompt_text, max_length=max_length, truncation=True)
    
    input_ids = full_tokens["input_ids"]
    labels = input_ids.copy()
    
    # Mask prompt tokens with -100 so loss is only on cypher
    prompt_length = len(prompt_tokens["input_ids"])
    labels[:prompt_length] = [-100] * prompt_length
    
    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels
    }

def get_tokenized_datasets(tokenizer, max_length=512):
    """Load and tokenize all splits."""
    train, val, test = load_splits()
    
    def tokenize(example):
        return tokenize_example(example, tokenizer, max_length)
    
    train_tokenized = train.map(tokenize, remove_columns=train.column_names)
    val_tokenized = val.map(tokenize, remove_columns=val.column_names)
    
    # Test set we don't tokenize for training — just return raw
    return train_tokenized, val_tokenized, test

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train, val, test = get_tokenized_datasets(tokenizer)
    
    print("Train size:", len(train))
    print("Val size:", len(val))
    print("Test size:", len(test))
    print("\nSample input_ids length:", len(train[0]['input_ids']))
    print("Sample labels (first 10):", train[0]['labels'][:10])
    print("Sample labels (last 10):", train[0]['labels'][-10:])