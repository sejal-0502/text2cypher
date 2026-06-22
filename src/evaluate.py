# evaluate.py
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data import format_prompt, MODEL_NAME

# def load_model(model_path):
#     """Load model and tokenizer from a given path or HF hub."""

#     print(f"Loading model from: {model_path}")
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
#     model.eval()
#     return model, tokenizer

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    
    # Handle subfolder format e.g. "username/repo/subfolder"
    parts = model_path.split("/")
    if len(parts) == 3:
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = parts[2]
        tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
        model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=subfolder, dtype=torch.float32)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    
    model.eval()
    return model, tokenizer

def generate_cypher(model, tokenizer, example, max_new_tokens=128):
    """Generate cypher for a single example."""
    
    # Format prompt WITHOUT the answer
    messages = format_prompt(example, include_answer=False)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # Greedy decoding — deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens (not the prompt)
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text.strip()

def exact_match(pred, ground_truth):
    """Check if prediction exactly matches ground truth."""

    return int(pred.strip() == ground_truth.strip())

def bleu_score(pred, ground_truth):
    """Compute sentence-level BLEU score."""

    pred_tokens = pred.strip().split()
    truth_tokens = ground_truth.strip().split()
    
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(
        [truth_tokens],
        pred_tokens,
        smoothing_function=smoothing
    )
    return round(score, 4)

def evaluate(model_path, output_file):
    """Run evaluation on test set and save results."""
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Load test set
    ds = load_dataset("RomanTeucher/text2cypher-curated")
    test = ds['test']
    
    results = []
    total_exact_match = 0
    total_bleu = 0
    
    print(f"Evaluating on {len(test)} test examples...")
    
    for example in tqdm(test):
        # Generate cypher
        generated = generate_cypher(model, tokenizer, example)
        ground_truth = example['cypher']
        
        # Compute metrics
        em = exact_match(generated, ground_truth)
        bleu = bleu_score(generated, ground_truth)
        
        total_exact_match += em
        total_bleu += bleu
        
        # Save per-sample result
        results.append({
            "instance_id": example['instance_id'],
            "question": example['question'],
            "schema": example['schema'],
            "ground_truth_cypher": ground_truth,
            "generated_cypher": generated,
            "exact_match": em,
            "bleu_score": bleu
        })
    
    # Aggregate scores
    n = len(test)
    aggregate = {
        "exact_match": round(total_exact_match / n, 4),
        "bleu_score": round(total_bleu / n, 4),
        "total_examples": n
    }
    
    # Save to file
    output = {
        "aggregate": aggregate,
        "results": results
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n--- Results ---")
    print(f"Exact Match: {aggregate['exact_match']}")
    print(f"BLEU Score:  {aggregate['bleu_score']}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME, help="Model path or HF hub name")
    parser.add_argument("--output", default="results/baseline_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    evaluate(args.model, args.output)